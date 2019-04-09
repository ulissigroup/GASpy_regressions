'''
This submodule is meant to house various functions we use to analyze our
database of results.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from collections import defaultdict
import random
import pickle
import numpy as np
from plotly import plotly
import plotly.graph_objs as go
from gaspy.utils import read_rc
from gaspy.gasdb import get_adsorption_docs


def create_gridplot(adsorbate, targets, filename, hovertext_labels=None):
    '''
    This function will create and save a gridplot of our adsorption energy
    data.

    Args:
        adsorbate           A string indicating which adsorbate you want to
                            plot the data for
        targets             A 2-tuple of floats indicating the low and high
                            range of the adsorption energies you consider to be
                            good, respectively.
        filename            A string indicating where you want to save the plot
                            (within the Plotly account)
        hovertext_labels    A sequence of strings indicating which
                            data you want displayed in the hovertext.
                            Possible strings include everything in the
                            `gaspy.defaults.adsorption_projections` dictionary
                            in addition to `stoichiometry` and `date`.
    Returns:
        url     The URL for the plot you just created
    '''
    # Python doesn't like mutable defaults
    if hovertext_labels is None:
        hovertext_labels = {'coordination', 'energy', 'stoichiometry', 'date',
                            'mpid', 'miller'}

    # Get and preprocess all the documents we have now
    extra_projections = {'atoms': '$atoms',
                         'date': '$calculation_dates.slab+adsorbate'}
    all_docs = get_adsorption_docs(adsorbate, extra_projections)
    with open(read_rc('gasdb_path') + 'mp_comp_data.pkl', 'rb') as file_handle:
        composition_by_mpid = pickle.load(file_handle)
    for doc in all_docs:
        doc['composition'] = composition_by_mpid[doc['mpid']]
    all_elements = {element for doc in all_docs
                    for element in composition_by_mpid[doc['mpid']]}

    # Organize all of our documents according to their bimetallic composition
    docs_by_comp = defaultdict(list)
    for doc in all_docs:
        comp = doc['composition']
        if len(comp) == 2:
            docs_by_comp[tuple(comp)].append(doc)
            docs_by_comp[tuple(reversed(comp))].append(doc)
        elif len(comp) == 1:
            docs_by_comp[tuple([comp[0], comp[0]])].append(doc)

    # Create the local coordinates for each set of bimetallics
    max_radius = 0
    for (element_i, element_j), docs in docs_by_comp.items():
        n = len(docs)
        width = np.sqrt(n)

        # Add `x` dimension to documents, which is uniform random sorted by
        # adsorption energy
        X = np.random.uniform(-width/2, width/2, n)
        X.sort()
        docs.sort(key=lambda doc: doc['energy'])
        for doc, x in zip(docs, X):
            doc['x'] = x

        # Add `y` dimension to documents, which is uniform random sorted by
        # composition
        Y = np.random.uniform(-width/2, width/2, n)
        Y.sort()
        for doc in docs:
            symbol_counts = doc['atoms']['symbol_counts']
            n_atoms = symbol_counts[element_i] + symbol_counts[element_j]
            ratio = symbol_counts[element_i] / n_atoms
            doc['ratio'] = ratio
            doc['stoichiometry'] = {element_i: symbol_counts[element_i],
                                    element_j: symbol_counts[element_j]}
        docs.sort(key=lambda doc: doc['ratio'])

        # Shuffle the y values within each ratio so that we get squares instead
        # of lines
        ratios = [doc['ratio'] for doc in docs]
        unique_ratios = sorted(list(set(ratios)))
        shuffle_counter = 0
        for i, ratio in enumerate(unique_ratios):
            ratio_count = ratios.count(ratio)
            ys = Y[shuffle_counter:shuffle_counter+ratio_count]
            random.shuffle(ys)
            shuffle_counter += ratio_count

        # Concatenate the appropriately shuffled uniform distribution with
        # documents
        for doc, y in zip(docs, Y):
            doc['y'] = y

        # Recalculate the size of the biggest square. We use this to scale
        # everything.
        max_radius = max([max_radius] + [max(doc['x'], doc['y']) for doc in docs])
    max_width = max_radius * 2

    # Settings for interactive image
    marker_size = 4
    font_size = 24
    font = dict(family='Arial', color='black')
    width = 900
    height = 800
    axis_font_size = 12

    # Set thresholds for energy
    low_energy = targets[0]
    high_energy = targets[1]
    good_energy = (low_energy + high_energy) / 2

    # We need the max and min energies to make sure the color mapping in all
    # our squares map to each other
    all_energies = [doc['energy'] for doc in all_docs]
    energy_min = min(all_energies)
    energy_max = max(all_energies)

    # Plotly lets you set colors only based on their normalized values, so we
    # need to normalize our energies before mapping colors onto them.
    energy_bandwidth = energy_max - energy_min
    low_energy_normalized = (low_energy - energy_min) / energy_bandwidth
    good_energy_normalized = (good_energy - energy_min) / energy_bandwidth
    high_energy_normalized = (high_energy - energy_min) / energy_bandwidth

    # Make our colorscale
    low_color = 'rgb(0, 0, 0)'
    good_color = 'rgb(175, 0, 255)'
    high_color = 'rgb(255, 200, 200)'
    colorscale = [(0., low_color),
                  (low_energy_normalized, low_color),
                  (good_energy_normalized, good_color),
                  (high_energy_normalized, high_color),
                  (1., high_color)]

    # Sort the elements according to how many good calculations we have
    n_calcs_by_element = defaultdict(int)
    for element_i in all_elements:
        for element_j in all_elements:
            docs = [doc for doc in docs_by_comp[element_i, element_j]
                    if low_energy <= doc['energy'] <= high_energy]
            n_calcs_by_element[element_i] += len(docs)
    elements_sorted = [element for element, count in
                       sorted(n_calcs_by_element.items(),
                              key=lambda kv: kv[1],
                              reverse=True)]

    # Figure out the spacings between each square in the grid
    traces = []
    for i, element_i in enumerate(elements_sorted):
        x_offset = (i+1) * max_width
        for j, element_j in enumerate(elements_sorted):
            y_offset = (j+1) * max_width

            # If we have an empty square, move on
            try:
                docs = docs_by_comp[(element_i, element_j)]
            except KeyError:
                continue

            # Get all the data out of the documents
            Xs = np.array([doc['x'] for doc in docs]) + x_offset
            Ys = np.array([doc['y'] for doc in docs]) + y_offset
            energies = [doc['energy'] for doc in docs]
            hovertexts = [doc_to_hovertext(doc, hovertext_labels) for doc in docs]

            # Make the graphical object traces for each data set, along with
            # all of the appropriate formatting
            trace = go.Scattergl(x=Xs, y=Ys,
                                 mode='markers',
                                 marker=dict(size=marker_size,
                                             color=energies,
                                             colorscale=colorscale,
                                             cmin=energy_min,
                                             cmax=energy_max),
                                 text=hovertexts)
            traces.append(trace)

    # Add a trace for the colorbar
    trace = go.Scattergl(x=[0, 0], y=[0, 0],
                         mode='markers',
                         marker=dict(size=0.1,
                                     color=[energy_min, energy_max],
                                     colorscale=[(0., low_color),
                                                 (0.5, good_color),
                                                 (1., high_color)],
                                     cmin=low_energy,
                                     cmax=high_energy,
                                     showscale=True),
                         hoverinfo=None)
    traces.append(trace)

    # Format the x and y axes
    axes_labels = dict(ticks='',
                       tickmode='array',
                       tickvals=np.linspace(max_width,
                                            len(all_elements)*max_width,
                                            len(all_elements)),
                       ticktext=[element for element in elements_sorted],
                       tick0=max_width/2,
                       dtick=max_width,
                       tickfont=dict(size=axis_font_size),
                       showgrid=False)

    # Format the plot itself
    layout = go.Layout(title=filename.split('/')[-1],
                       titlefont=font,
                       xaxis=axes_labels,
                       yaxis=axes_labels,
                       showlegend=False,
                       width=width, height=height,
                       font=dict(size=font_size))

    # Save it online
    plotly.sign_in(**read_rc('plotly_login_info'))
    url = plotly.plot(go.Figure(data=traces, layout=layout), filename=filename)
    return url


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
