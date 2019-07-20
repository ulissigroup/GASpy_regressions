'''
This submodule is meant to create plots to analyze our
database of adsorption results 2-dimentionally.
'''

__author__ = 'Aini Palizhati'
__email__ = 'apalizha@andrew.cmu.edu'

import numpy as np
import pandas as pd
import pickle
import tqdm
import warnings
warnings.filterwarnings('ignore')

from plotly import plotly
import plotly.graph_objs as go
from pymatgen.ext.matproj import MPRester
from gaspy.utils import read_rc
from gaspy import gasdb
from gaspy.mongo import make_atoms_from_doc
# from gaspy_regress import awscache
plotly.plotly.sign_in(**read_rc('plotly_login_info'))

def plot_2D_plot(adsorbate1, adsorbate2, adsorbate1_correction, adsorbate2_correction):
    """
    When called, this function will return a 2D plotly plot with x-aixs
    being adsorbate1, y-axis being adsorbate2.
    In addition, it will also cache the adsorption configuration to AWS.
    The plot can be accessed via
        http://ulissigroup.cheme.cmu.edu/gaspy_plots/index.html?plotly=(plot# here)

    Args:
        adsorbate1,     adsorbate1, who's energy is going on x-axis
        adsorbate1,     adsorbate2, who's energy is going on y-axis
        adsorbateX_correction: energy correction for adsorbate on X-axis
                                 e.g. we have dE for adsorption energy, if
                                 we want to convert to dG.
                                 default is 0 if we plot dE
          adsorbateY_correction: energy correction for adsorbate on Y-axis
                                 e.g. we have dE for adsorption energy, if
                                 we want to convert to dG.
                                 default is 0 if we plot dE

    """
    grouped_df = _prepare_grouped_df(adsorbate1, adsorbate2)

    puremetal_mpid = ['mp-101', 'mp-23', 'mp-2', 'mp-91', 'mp-1080711', 'mp-1061133', 'mp-58', 'mp-13', 'mp-124', 
                  'mp-30', 'mp-32', 'mp-72', 'mp-142', 'mp-81', 'mp-79', 'mp-1056037', 'mp-94', 'mp-126',
                  'mp-146', 'mp-1055994', 'mp-104', 'mp-117', 'mp-35', 'mp-90', 'mp-75', 'mp-10172',
                  'mp-632250', 'mp-1056438', 'mp-8', 'mp-74', 'mp-20483', 'mp-54', 'mp-48', 'mp-134',
                  'mp-129', 'mp-11', 'mp-1056486', 'mp-149', 'mp-45', 'mp-49', 'mp-96', 'mp-85',
                  'mp-568584', 'mp-754514', 'mp-33', 'mp-999498', 'mp-127', 'mp-25', 'mp-132', 'mp-570747',
                  'mp-14', 'mp-154', 'mp-672234', 'mp-672233']

    # DFT_adsorbate1, DFT_adsorbate2
    DFT_ads1_ads2 = grouped_df.loc[(grouped_df['{}_DFT'.format(adsorbate1)] == True)&(grouped_df['{}_DFT'.format(adsorbate2)] == True)]
    metal_DFT_ads1_ads2 =  DFT_ads1_ads2.loc[DFT_ads1_ads2['mpid'].isin(puremetal_mpid)]
    intermetallics_DFT_ads1_ads2 = DFT_ads1_ads2[~DFT_ads1_ads2.isin(metal_DFT_ads1_ads2)].dropna()

    # DFT_adsorbate1, ML_adsorbate2
    DFT_ads1_ML_ads2 = grouped_df.loc[(grouped_df['{}_DFT'.format(adsorbate1)] == True)&(grouped_df['{}_DFT'.format(adsorbate2)] == False)]
    metal_DFT_ads1_ML_ads2 = DFT_ads1_ML_ads2.loc[DFT_ads1_ML_ads2['mpid'].isin(puremetal_mpid)]
    intermetallics_DFT_ads1_ML_ads2 = DFT_ads1_ML_ads2[~DFT_ads1_ML_ads2.isin(metal_DFT_ads1_ML_ads2)].dropna()

    # ML_adsorbate1, DFT_adsorbate2
    ML_ads1_DFT_ads2 = grouped_df.loc[(grouped_df['{}_DFT'.format(adsorbate1)] == False)&(grouped_df['{}_DFT'.format(adsorbate2)] == True)]
    metal_ML_ads1_DFT_ads2 = ML_ads1_DFT_ads2.loc[ML_ads1_DFT_ads2['mpid'].isin(puremetal_mpid)]
    intermetallics_ML_ads1_DFT_ads2 = ML_ads1_DFT_ads2[~ML_ads1_DFT_ads2.isin(metal_ML_ads1_DFT_ads2)].dropna()

    # get scatter points for plotting
    print('Plotting DFT_{}, DFT_{}'.format(adsorbate1, adsorbate2))
    # DFT_adsorbate1, DFT_adsorbate2
    data = _make_scatter_points(metal_DFT_ads1_ads2, adsorbate1, adsorbate2,
                                'monometallic DFT {} & {}'.format(adsorbate1, adsorbate2),
                                'square', 'red', 'red',
                                adsorbate1_correction, adsorbate2_correction)
    data += _make_scatter_points(intermetallics_DFT_ads1_ads2, adsorbate1, adsorbate2,
                                 'intermetallic DFT {} & {}'.format(adsorbate1, adsorbate2),
                                 'circle', 'white', 'red',
                                 adsorbate1_correction, adsorbate2_correction)

    # DFT_adsorbate1, ML_adsorbate2
    print('Plotting DFT_{}, ML_{}'.format(adsorbate1, adsorbate2))
    data += _make_scatter_points(metal_DFT_ads1_ML_ads2, adsorbate1, adsorbate2,
                                 'monometallic DFT {} & ML {}'.format(adsorbate1, adsorbate2),
                                 'square', 'yellowgreen', 'yellowgreen',
                                 adsorbate1_correction, adsorbate2_correction)
    data += _make_scatter_points(intermetallics_DFT_ads1_ML_ads2, adsorbate1, adsorbate2,
                                 'intermetallic DFT {} & ML {}'.format(adsorbate1, adsorbate2),
                                 'circle', 'white', 'yellowgreen',
                                 adsorbate1_correction, adsorbate2_correction)

    # ML_adsorbate1, DFT_adsorbate2
    print('Plotting ML_{}, DFT_{}'.format(adsorbate1, adsorbate2))
    data += _make_scatter_points(metal_ML_ads1_DFT_ads2, adsorbate1, adsorbate2,
                                 'monometallic ML {} & DFT {}'.format(adsorbate1, adsorbate2),
                                 'square', 'cornflowerblue', 'cornflowerblue',
                                 adsorbate1_correction, adsorbate2_correction)
    data += _make_scatter_points(intermetallics_ML_ads1_DFT_ads2, adsorbate1, adsorbate2,
                                 'intermetallic ML {} & DFT {}'.format(adsorbate1, adsorbate2),
                                 'circle', 'white', 'cornflowerblue',
                                 adsorbate1_correction, adsorbate2_correction)

    if adsorbate1_correction == 0 and adsorbate2_correction == 0:
        fig = go.Figure(data=data,layout=go.Layout(hovermode='closest',
                                                   xaxis=dict(title='dE_{} [eV]'.format(adsorbate1), titlefont=dict(size=25)),
                                                   yaxis=dict(title='dE_{} [eV]'.format(adsorbate2), titlefont=dict(size=25))))

    elif adsorbate1_correction != 0 and adsorbate2_correction != 0:
        fig = go.Figure(data=data, layout=go.Layout(hovermode='closest',
                                                   xaxis=dict(title='dG_{} [eV]'.format(adsorbate1), titlefont=dict(size=25)),
                                                   yaxis=dict(title='dG_{} [eV]'.format(adsorbate2), titlefont=dict(size=25))))
    else:
        print('you only added correction to 1 of the adsorbate energies. please keep the energies consistent')

    print('Your figure location is in Plotly 2D_plots/{}_{}_bySurface'.format(adsorbate1, adsorbate2))
    plotly.iplot(fig, filename='2D_plots/{}_{}_bySurface'.format(adsorbate1, adsorbate2))


def _make_df_from_docs(adsorbate, columns_name):
    """
    This function takes the adsorbate of interest
    and pull low_coverage_docs. The docs are used
    to make dataframe that will be used for making 2D plots.

    Args:
        adsorbate       adsorbate of interest
        column_names    A list contains column names of the dataframe
                        (excluding mong_id, coordination, adsorbate,
                         and if it's DFT caluclations)

    Returns:
        adsorbate_df    dataframe of that adsorbate
    """
    adsorbate_docs = gasdb.get_low_coverage_docs(adsorbate)

#     # Add atoms & results into each docs
#     # Cache the adsorption configurations
#     print('Caching adsorption configurations')
#     for doc in adsorbate_docs :
#         if doc['DFT_calculated'] == True:
#             with gasdb.get_mongo_collection(collection_tag='adsorption') as collection:
#                 mongo_doc = list(collection.find({'_id':doc['mongo_id']}))
#         else:
#             with gasdb.get_mongo_collection(collection_tag='catalog') as collection:
#                 mongo_doc = list(collection.find({'_id':doc['mongo_id']}))
#         doc['atoms'] = mongo_doc['atoms']
#         doc['results'] = mong_doc['results']
#     awscache.cache_docs_to_images(adsorbate_docs, 4)

    # take the docs and extract information (mpid, miller, top,
    # shift, mongo_id, coordination, DFT_calculated(if it's DFT calculation)
    docs_info = [[doc['mpid'], doc['miller'][0], doc['miller'][1], doc['miller'][2],
                  doc['top'], doc['shift'], doc['mongo_id'], doc['coordination'],
                  doc['energy'], doc['DFT_calculated']] for doc in adsorbate_docs]

    # Create a list of column names and make a dataframe.
    # Sort this dataframe by surface & adsorption energy and
    # get the minimum energy on each surface
    column_name_final = columns_name + ['{}_mongo'.format(adsorbate),
                                        '{}_coordination'.format(adsorbate),
                                        '{}'.format(adsorbate),
                                        '{}_DFT'.format(adsorbate)]
    dataframe = pd.DataFrame(docs_info, columns=column_name_final).sort_values(by=columns_name + [adsorbate])
    dataframe = dataframe.drop_duplicates(subset=columns_name, keep='first')
    return dataframe


def _prepare_grouped_df(adsorbate1, adsorbate2):
    """
    This function takes the adsorbates of interest
    and make a dataframe that will be used for making 2D plots.
    Each row in the dataframe is grouped by unique surface.
    A surface is defined with mpid, Miller index, top, and shift.

    Args:
        adsorbate1      adsorbate1, who's energy is going on x-axis
        adsorbate1      adsorbate2, who's energy is going on y-axis

    Returns:
        adsorbate1_df   dataframe of adsorbate1
        adsorbate2_df   dataframe of adsorbate2
        grouped_df      dataframe used for plotting
    """
    surface_fp = ['mpid', 'millerx', 'millery', 'millerz', 'top', 'shift']
    adsorbate1_df = _make_df_from_docs(adsorbate1, surface_fp)
    adsorbate2_df = _make_df_from_docs(adsorbate2, surface_fp)

    # merge them together based on unique surface
    grouped_results = pd.merge(adsorbate1_df, adsorbate2_df, on=surface_fp).dropna()
    # drop rows that has ML prediction on both OH & CO
    grouped_results = grouped_results.drop(grouped_results[(grouped_results['{}_DFT'.format(adsorbate1)] == False) & (grouped_results['{}_DFT'.format(adsorbate2)] == False)].index).reset_index()

    # Add formula to the dataframe based on mpid
    rc = read_rc()
    atoms_db = gasdb.get_mongo_collection('atoms')
    mpids = set(grouped_results['mpid'])
    compositions_by_mpid = {}
    print('Beginning to pull data from the Materials Project...')
    with MPRester(read_rc()['matproj_api_key']) as mat_proj:
        for mpid in tqdm.tqdm_notebook(mpids):
            try:
                entry = mat_proj.get_entry_by_material_id({'task_ids': mpid})
                compositions_by_mpid[mpid] = entry.composition.get_reduced_formula_and_factor()[0]
            except IndexError:
                compositions_by_mpid[mpid] = ""
    data = list(compositions_by_mpid.items())
    df_new = pd.DataFrame(data, columns=['mpid', 'formula'])
    grouped_df = pd.merge(grouped_results, df_new, on='mpid')


    return grouped_df

def doc_to_hovertext(doc, hovertext_labels):
    '''
    Make a function to pull text out of a document and turn it into
    hovertext.

    Args:
        doc                 A dictionary that you got from Mongo
        hovertext_labels    The keys in the dictionary you want parsed
    Returns:
        text    The parsed version of the document that you can pass
                to plotly as hovertext
    '''
    text = ''
    for label, fp_value in doc.items():
        if label in hovertext_labels:
            text += '<br>' + str(label) + ':  ' + str(fp_value)
    return text

def _make_scatter_points(dataframe, adsorbateX, adsorbateY, category, scatter_shape, scatter_color, line_color, adsorbateX_correction, adsorbateY_correction):
    """
    Helper function that takes dataframe to plot the different dataframe,
    to make the repeating process easier.

    Args:
          dataframe              dataframe used for plotting
          adsorbateX             adsorbate on x-axis
          adsorbateY             adsorbate on y-axis
          category               What category of data is it
                                 e.g. pure_metal DFT_adsorbate1, ML_adsorbate2
                                      intermetallic ML adsorbate 1, DFT adsorbate2
          scatter_shape          shape of the scatters
          scatter_color          color of the scatters
          line_color             the color of the scatter shapes edge
          adsorbateX_correction  energy correction for adsorbate on X-axis
                                 e.g. we have dE for adsorption energy, if
                                 we want to convert to dG.
                                 default is 0 if we plot dE
          adsorbateY_correction  energy correction for adsorbate on Y-axis
                                 e.g. we have dE for adsorption energy, if
                                 we want to convert to dG.
                                 default is 0 if we plot dE
    Returns:
        data                     Data of the scatter points
    """
    # define what we want to display when hovering the scatter points
    display_fps = set(['mpid', 'formula', 'miller', 'top', 'shift', '{}_coordination'.format(adsorbateX), '{}_coordination'.format(adsorbateY)])

    if len(dataframe) != 0:
        all_docs = []
        for index, row in dataframe.iterrows():
            doc = {}
            doc['mpid'] = row['mpid']
            doc['miller'] = [row['millerx'], row['millery'], row['millerz']]                                
            doc['shift'] = round(row['shift'],3)
            doc['top'] = row['top']
            doc['formula'] = "".join(row['formula'])
            doc['{}_coordination'.format(adsorbateX)] = row['{}_coordination'.format(adsorbateX)]
            doc['{}_coordination'.format(adsorbateY)] = row['{}_coordination'.format(adsorbateY)]
            all_docs.append(doc)
        infos = [[doc_to_hovertext(doc, display_fps)] for doc in all_docs]

        X = np.array(dataframe[['{}'.format(adsorbateX)]]+adsorbateX_correction)
        Y = np.array(dataframe[['{}'.format(adsorbateY)]]+adsorbateY_correction)
        zipped_results = list(zip(infos, X, Y))
        info, x, y = zip(*zipped_results)

        data = [go.Scatter(x=x, y=y,
                           customdata=[{'mongo_id_xaxis':str(a), 'mongo_id_yaxis':str(b)} for a, b in zip(dataframe['{}_mongo'.format(adsorbateX)],dataframe['{}_mongo'.format(adsorbateY)])],
                           mode='markers',
                           name=category,
                           text=[a[0] for a in info],
                           marker={'size': 8,
                                   'symbol': scatter_shape,
                                   'color': scatter_color,
                                   'opacity': 1.0,
                                   'line': {
                                       'color' : line_color,
                                       'width' : 2}})]
    else:
        data = []
    return data
