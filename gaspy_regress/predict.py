'''
This module contains functions that apply transformations to the simulation predictions created by
GASpy and the the surrogate model estimations created by GASpy_regressions.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb  # noqa: F401
import time
import pickle
import json
import numpy as np
import pandas as pd
import tqdm
from gaspy import utils, gasdb, defaults   # noqa: E402


def volcano(regressor, regressor_block, sheetname, excel_file_path, scale,
            adsorbate, fp_blocks='default', descriptor='energy',
            processes=32, doc_chunk_size=1000, save_all_estimates=False,
            vasp_settings=None, energy_min=-4, energy_max=4, f_max=0.5,
            ads_move_max=1.5, bare_slab_move_max=0.5, slab_move_max=1.5):
    '''
    Map both simulation results and surrogate model results onto a volcano relationship.
    Note that this function uses the name `x` to denote the simulation results and
    surrogate model results (e.g., adsorption energies); it also uses the name `y`
    to denote the post-volcano-transformations (e.g., activity, selectivity).

    Inputs:
        regressor           An instance of a fitted GASpyRegressor object
        regressor_block     If your regression model is blocked, you'll need to specify
                            which block you want to use to make the predictions. This
                            will probably be a tuple of strings, like ('CO',). If there
                            was no block, then use (None,)
        These inputs define the volcano you want to use to transform the data:
            sheetname       A string indicating the name of the Excel worksheet that
                            contains the volcano information. AKA `system` in some other
                            parts of GASpy_regressions
            excel_file_path A string indicating the location of the properly-formatted
                            excel file that contains the incumbent volcano
            scale           A string indicating the scale across which the volcano curve
                            varies (e.g., linear, logarithmic, etc.).
        adsorbate           A string indicating the adsorbate that you want to be studying.
                            If you want more than one adsorbate, then you should probably
                            make another function.
        fp_blocks           A list of fingerprints over which to find minima
                            (reference the `_minimize_over` function). If you don't want to
                            block at all, then you may set it to `None`
        descriptor          A string indicating the descriptor you are using to
                            create the volcano.
        processes          We do multiprocessing. This decides how many processes we use.
        doc_chunk_size      This integer argument decides how many documents a child process
                            should process at a time when doing ML estimations. Bigger chunks
                            yield faster runs, but are more prone to memory issues.
        save_all_estimates  A boolean indicating whether or not you want to save all of
                            the estimates we created with the regressor's `predict` method.
        All of the other inputs are filter settings for the data that we want
        to pull from the database of results.
    Outputs:
        sim_data    A list of nested tuples, structured as follows:
                    (mongo documents, simulations, estimations)
                    Where `mongo documents` are the fingerprints,
                    `simulations` are structured as:
                        (descriptor, transformation)
                        Where descriptor (e.g., energy) is:
                            (value, uncertainty)
                        and transformation (e.g., activity) is:
                            (value, uncertainty)
                    and where `estimations` are structured identicalaly to `simulations`
        unsim_data  Same as `sim_data`, but for surrogate-model-estimated
                    data instead.
    '''
    # Some of our defaults need to be lists, which are mutable. So we define them
    # down here.
    if fp_blocks == 'default':
        fp_blocks = ['mpid', 'miller', 'top']

    # Load the literature volcano data
    volcano, dE_expt, y_expt, label_expt = \
        _pull_literature_volcano(excel_file_path, sheetname, scale=scale)

    # Define the fingerprints to pull from the database, and add the descriptor
    # to make sure that we get it out.
    # Note that we filter the "bad" documents from the results database
    # by specifying a lot of filters.
    with gasdb.get_adsorption_client() as ads_client:
        fingerprints = defaults.fingerprints(simulated=True)
        ads_docs = gasdb.get_docs(client=ads_client, collection_name='adsorption',
                                  adsorbates=[adsorbate],
                                  fingerprints=fingerprints,
                                  vasp_settings=vasp_settings,
                                  energy_min=energy_min,
                                  energy_max=energy_max,
                                  f_max=f_max,
                                  ads_move_max=ads_move_max,
                                  bare_slab_move_max=bare_slab_move_max,
                                  slab_move_max=slab_move_max)
        ads_pdocs = utils.docs_to_pdocs(ads_docs)
        ads_dE = np.array(ads_pdocs[descriptor])

    # Get the whole catalog
    with gasdb.get_catalog_client() as cat_client:
        cat_docs = gasdb.get_docs(client=cat_client, collection_name='catalog')
    # Get indices to split the catalog into sim and unsim lists, then use the indices
    # to create the appropriate objects
    print('Finding the indices to split the catalog...')
    tic = time.time()
    sim_inds, unsim_inds = gasdb.split_catalog(ads_docs, cat_docs)
    toc = time.time()
    print('It took %i seconds to find the indices to split the catalog' % (toc-tic))

    # Catalog documents don't have any information about adsorbates. But if our model
    # requires information about adsorbates, then we probably need to put it in.
    # Note that we have an EAFP wrapper to check whether our model is hierarchical or not.
    try:
        features = regressor.features + regressor.features_inner
    except AttributeError:
        features = regressor.features
    if 'ads' in features:
        for doc in cat_docs:
            doc['adsorbate'] = doc['adsorbates'][0]
    # Create the regressor's estimations
    print('Starting catalog estimations...')
    cat_dE = regressor.predict(cat_docs, block=regressor_block,
                               processes=processes,
                               doc_chunk_size=doc_chunk_size)

    # Use the splitting indices we found earlier to split our data.
    # `unsim_*` objects represent configurations that we never simulated (ML-only).
    # `ads_*_est` objects represent ML estimations of configurations that we have simulated.
    # `ads_*` objects represent actual simulation results.
    print('Splitting all of the simulated and unsimulated data (will see 3 progress bars)...')
    unsim_cat_docs = [cat_docs[i] for i in tqdm.tqdm(unsim_inds, total=len(unsim_inds))]
    unsim_cat_dE = [cat_dE[i] for i in tqdm.tqdm(unsim_inds, total=len(unsim_inds))]
    ads_dE_est = [cat_dE[i] for i in tqdm.tqdm(sim_inds, total=len(sim_inds))]

    # Add the adsorption energies into the documents. And add tags to tell us whether
    # the dE came from ML or not.
    print('Adding extra tags to the documents for future reference...')
    tic = time.time()
    for doc, dE in zip(ads_docs, ads_dE):
        doc['energy'] = dE
        doc['ML'] = False
    for doc, dE in zip(unsim_cat_docs, unsim_cat_dE):
        doc['energy'] = dE
        doc['ML'] = True
    # Combine all the documents into one list. Then make it json compatible.
    all_docs = ads_docs + unsim_cat_docs
    for doc in all_docs:
        doc['mongo_id'] = str(doc['mongo_id'])
    toc = time.time()
    print('It took %i seconds to add the extra tags' % (toc-tic))

    # Optional snippet to save all of our predictions
    if save_all_estimates:
        print('Saving all of the estimates...')
        tic = time.time()
        gaspy_path = utils.read_rc('gaspy_path')
        save_folder = gaspy_path + '/GASpy_regressions/pkls/'
        with open(save_folder + 'all_estimates_for_%s.json' % adsorbate, 'w') as f:
            json.dump(all_docs, f)
        toc = time.time()
        print('It took %i seconds to save all the estimates' % (toc-tic))

    # Filter the data over each fingerprint block, as per the `_minimize_over` function.
    if fp_blocks:
        print('Starting minimize_over...')
        tic = time.time()
        cat_docs, cat_dE, ads_docs, ads_dE = \
            _minimize_over(unsim_cat_docs, unsim_cat_dE, ads_docs, ads_dE, fp_blocks)
        toc = time.time()
        print('It took %i seconds to minimize_over' % (toc-tic))

    # Re-perform the ML estimations of the [filtered,] simulated configurations.
    # We've technically already done them, but this is the easiest way to code
    # this. If we start running into scaling issues, then we can figure out how
    # to parse this information out instead of re-calculation.
    print('Re-calculating the [filtered,] simulated configurations...')
    ads_dE_est = regressor.predict(ads_docs, block=regressor_block,
                                   processes=processes,
                                   doc_chunk_size=doc_chunk_size)

    # Transform the volcano x-axis into the y-axis. We use multiprocessing for this
    # transformation. If you find a memory issue, then you should set chunksize explicitly.
    print('Starting triple volcano-ing...')
    def calc_volcano(input_energies):  # noqa: E306
        chunksize = len(input_energies) / processes / 10
        # We make sure chunksize is at least 1 to avoid zero division errors
        if chunksize < 1:
            chunksize = 1
        output = utils.multimap(volcano, input_energies,
                                processes=processes,
                                chunksize=chunksize,
                                n_calcs=len(input_energies)/chunksize)
        return np.array(output).flatten()
    cat_y = calc_volcano(cat_dE)
    ads_y = calc_volcano(ads_dE)
    ads_y_est = calc_volcano(ads_dE_est)

    # We also save the uncertainties of each of these values [eV].
    # Uncertainties are affixed with `u`.
    print('Starting data packaging...')
    tic = time.time()
    sim_u = 0.1  # simulation uncertainty
    model_u = regressor.rmses[regressor_block]['train']  # model uncertainty
    est_u = np.sqrt(sim_u**2 + model_u**2)  # total uncertainty of surrogate model
    ads_dE_u = [sim_u]*len(ads_dE)
    ads_y_u = [0.]*len(ads_y)
    ads_dE_est_u = [est_u]*len(ads_dE_est)
    ads_y_est_u = [0.]*len(ads_y_est)
    cat_dE_u = [est_u]*len(cat_dE)
    cat_y_u = [0.]*len(cat_y)

    # Zip up all of the information about simulated systems
    sim_data = zip(ads_docs,
                   zip(zip(ads_dE, ads_dE_u), zip(ads_y, ads_y_u)),
                   zip(zip(ads_dE_est, ads_dE_est_u), zip(ads_y_est, ads_y_est_u)))
    # Zip up all of the information about unsimulated, catalog systems
    unsim_data = zip(cat_docs,
                     zip(zip(cat_dE, cat_dE_u), zip(cat_y, cat_y_u)))
    toc = time.time()
    print('Took %i seconds to package' % (toc-tic))

    return sim_data, unsim_data


def best_surfaces(data_ball, performance_threshold, max_surfaces=1000):
    '''
    Turns a ball of data created by another `predict.*` function, such as `volcanos`,
    and parses it to create a list of the highest performing surfaces.

    Input:
        data_ball               Either a string indicating the location of the the data ball,
                                or the actual data ball.
        performance_threshold   A float (between 0 and 1, preferably) that indicates
                                the minimum level of performance relative to the best
                                performing surface.
        max_surfaces            An integer that sets the limit for how many surfaces
                                you want to report.
    Output:
        best_surfaces   A list of tuples containing the information for the best surfaces
        labels          A tuple of strings indicating what the data in `best_surfaces` refer to
    '''
    # Open the data ball if the user supplied a path
    if isinstance(data_ball, str):
        with open(data_ball, 'r') as f:
            data_ball = pickle.load(f)
    # Unpack the databall
    sim_data, unsim_data = data_ball
    sim_docs, predictions, _ = zip(*sim_data)
    cat_docs, estimations = zip(*unsim_data)
    x_data_pred, y_data_pred = zip(*predictions)
    x_pred, x_u_pred = zip(*x_data_pred)
    y_pred, y_u_pred = zip(*y_data_pred)
    x_data_est, y_data_est = zip(*estimations)
    y_est, y_u_est = zip(*y_data_est)
    x_est, x_u_est = zip(*x_data_est)

    # Package the estimations and predictions together, because we don't
    # really care which they come from. Then zip it up so we can sort everything
    # at once.
    docs = list(sim_docs)
    docs.extend(list(cat_docs))
    x = list(x_pred)
    x.extend(list(x_est))
    x_u = list(x_u_pred)
    x_u.extend(list(x_u_est))
    y = list(y_pred)
    y.extend(list(y_est))
    y_u = list(y_u_pred)
    y_u.extend(list(y_u_est))
    data = zip(docs, x, x_u, y, y_u)

    # Sort the data so that the items with the highest `y` values are
    # at the beginning of the list
    data = sorted(data, key=lambda datum: datum[3], reverse=True)
    # Take out everything that hasn't performed well enough, and trim some
    # more rows if our data set exceeds the threshold
    y_max = data[0][1]
    data = [(doc, _x, _x_u, _y, _y_u) for doc, _x, _x_u, _y, _y_u in data
            if _y > performance_threshold*y_max]
    if len(data) > max_surfaces:
        del data[max_surfaces+1:]
    # Find the best performing surfaces and pull out information that we want to pass along
    best_surfaces = []
    for doc, _x, _x_u, _y, _y_u in data:
        # Chemical formula (and a bunch of parsing to get rid of markers, adsorbates,
        # and to simply/reduce stoichiometry)
        formula = doc['formula']
        formula = formula.replace('U', '')
        # TODO:  Finish eliminating the adsorbate and fixing the stoichiometry
        # elements = re.findall('[A-Z][^A-Z]*', formula)
        # for i, element in enumerate(elements):
        #     el, num = re.split('(\d+)', element)
        # Material information
        mpid = doc['mpid']
        miller = tuple(doc['miller'])
        top = doc['top']
        mongo_id = doc['mongo_id']
        # Performance metrics
        energy = _x
        performance = _y
        surface = (mpid, formula, miller, top, energy, performance, mongo_id)
        best_surfaces.append(surface)
    # Define the labels
    labels = ('MPID', 'Formula', 'Miller', 'Top?', 'dE [eV]', 'Performance', 'Mongo ID')

    return best_surfaces, labels


def _minimize_over(cat_docs, cat_values, sim_docs, sim_values, fp_blocks):
    '''
    In some cases, we do not want to plot all of our data. We would rather
    plot only data that represent minima within certain blocks or fingerprints.
    want to find the minimum adsorption energies for a given surface for each
    bulk material we are looking at. This function performs this filtering for us.

    Inputs:
        cat_docs    A list of mongo documents of the catalog
        cat_values  A list of the catalog values that we want to use to minimize over
        sim_docs    A list of mongo documents of the simulations
        sim_values  A list of the simulation values that we want to use to minimize over
        fp_blocks   A list of strings for the fingerprints that you want to block within
    Outputs:
        cat_docs    A list of mongo documents of the catalog,
                    but with non-minima filtered out
        cat_values  A list of the catalog values that we want to use to minimize over,
                    but with non-minima filtered out
        sim_docs    A list of mongo documents of the simulations,
                    but with non-minima filtered out
        sim_values  A list of the simulation values that we want to use to minimize over,
                    but with non-minima filtered out
    '''
    # We add a new "fingerprint" to the docs so that we can keep track of
    # where data come from where after they've been pooled
    for i, doc in enumerate(cat_docs):
        doc['catalog?'] = True
        cat_docs[i] = doc
    for i, doc in enumerate(sim_docs):
        doc['catalog?'] = False
        sim_docs[i] = doc

    # Pool the catalog and simulation results together before finding the minima.
    # This ensures that each block will only have one data point show up on the plot.
    docs = cat_docs + sim_docs
    values = np.concatenate((cat_values, sim_values), axis=0)

    # We're going to parse our data into a dictionary, `block_data`, whose keys
    # will be tuples that represent the unique blocks of fingerprints,
    # e.g., ('mpid-23', '[2, 1, 1]'), and whose values will be lists of datum
    # that fall into that bucket. The "datum" will be 2-tuples whose first
    # element is the index of the datum within `values`, and whose second element
    # is the actual `value` from `values`.
    block_data = {}
    for i, (doc, value) in enumerate(zip(docs, values)):
        # Note that we turn the fingerprints into strings to make sure iterables
        # (like lists for miller indices) don't start doing funny things to our code.
        block = tuple([str(doc[fingerprint]) for fingerprint in fp_blocks])
        # EAFP to either append a new entry to an existing block, or create a new block
        try:
            block_data[block].append((i, value))
        except KeyError:
            block_data[block] = [(i, value)]

    # Now that our data is divided into blocks, we can start figuring out
    # which datum within each block's data set yields the minimum `value`.
    # We will then add the index of that datum to the `indices` set, which
    # we will use to rebuild/filter our data set.
    indices = set()
    for block, data in block_data.iteritems():
        min_block_index = np.argmin([value for (_, value) in data])
        indices.add(data[min_block_index][0])

    # Now create the inputs to create the outputs
    min_docs = [docs[i] for i in indices]
    min_values = [values[i] for i in indices]

    # Now un-pool
    cat_docs = []
    cat_values = []
    sim_docs = []
    sim_values = []
    for doc, value in zip(min_docs, min_values):
        if doc['catalog?']:
            cat_docs.append(doc)
            cat_values.append(value)
        else:
            sim_docs.append(doc)
            sim_values.append(value)

    return cat_docs, cat_values, sim_docs, sim_values


def _pull_literature_volcano(excel_file_path, sheetname, scale):
    '''
    This function pulls data from an Excel file to create a "volcano function",
    which is a function that turns an input x-value into an output y-value
    (e.g., adsorption energy to activity). It also outputs the experimental
    data points that should be in the same Excel file.

    Note that we've hard-coded the location of particular information within
    the Excel file, which means that you need to format the Excel file
    appropriately. Follow the template in the repository.

    Inputs:
        excel_file_path     A string indicating the location of the properly-formatted
                            excel file that contains the incumbent volcano
        sheetname           A string indicating the sheet within the Excel file that
                            the information in stored in
        scale               A string indicating the scale across which the volcano curve
                            varies (e.g., linear, logarithmic, etc.).
    Return:
        volcano A function whose input is a float (or a numpy.array) indicating
                the x-axis location on the volcano and whose output
                is a float (or a numpy.array) indicating the y-axis
                location.
        x       The x-values of the experimental data points
        y       The y-values of the experimental data points
        labels  The string-formatted labels of the experimental data points
    '''
    # Pull the dataframe out of Excel
    df = pd.read_excel(excel_file_path, sheetname=sheetname)
    # Pull out the coordinates (x, y) and the labels (labels) of the
    # experimental data points. We also filter out any extra `nan` entries
    # that may have appeared.
    labels = df.iloc[:, 0].get_values()
    y = df.iloc[:, 1].get_values()
    x = df.iloc[:, 2].get_values()
    labels = labels[~pd.isnull(labels)]
    x = x[~pd.isnull(x)]
    y = y[~pd.isnull(y)]
    # Do some fancy footwork to find `zenith`, which is the x-value at
    # the zenith of the volcano curve.
    zi = (df.iloc[:, 3] == 'Zenith')
    zenith = df.iloc[:, 2][zi].get_values()[0]
    # Find the slope and intercepts of the lines for both the LHS and
    # RHS of the volcano. Note that this is hard-coded, so make sure
    # the Excel file was completed as per the template.
    lhs_slope = df.iloc[0, 6]
    lhs_intercept = df.iloc[0, 7]
    rhs_slope = df.iloc[0, 10]
    rhs_intercept = df.iloc[0, 11]

    # All of our volcanos are 2-part functions. This `unpack` function returns
    # the parameters that are appropriate for whichever side of the volcano
    # we're on.
    def unpack(x):
        '''
        Unpack either the left-hand-side or right-hand-side parameters from the `parameters`
        dictionary, depending on whether the x-value is to the left or right of the zenith.
        '''
        m = np.where(x < zenith, lhs_slope, rhs_slope)
        b = np.where(x < zenith, lhs_intercept, rhs_intercept)
        return m, b

    # Create the volcano function assuming it's linear
    if scale == 'linear':
        def volcano(x):
            ''' Linear volcano '''
            m, b = unpack(x)
            return m*x + b

    # Create the volcano function assuming it's logarithmic
    elif scale == 'log':
        def volcano(x):
            ''' Exponential volcano '''
            m, b = unpack(x)
            return np.exp(m*x + b)

    else:
        raise Exception('You have not yet defined the function for a %s scale' % scale)

    return volcano, x, y, labels
