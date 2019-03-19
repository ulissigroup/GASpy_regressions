"""
This script contains helper functions that use Delta Learning Method to predict adsorption energy of OOH.
The overall steps are below:
    1. Group OH and OOH adsorption energy based on adsorption site
    2. Perform a linear regression on all the data points and obtain slope and intercept.
    3. Use TPOTRegressor to that predicts residual of each data point (residual deviated from the linear regression line), given fingerprints       (of the adsorption site).
        The hybrid machine learning model is dE(OOH) = slope*dE(OH)+intercept +ML correction term
    4. Pull Kevin's model0 prediction on OH adsorption energy 
    5. Use OH value & scalingML to predict OOH value
"""

import sys
sys.path.append('/project/projectdirs/m2755/software/')

from gaspy import gasdb, defaults
import warnings
warnings.filterwarnings('ignore')
from gaspy_regress import fingerprinters
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from tpot import TPOTRegressor
from sklearn.preprocessing import StandardScaler
import copy

def process_adsorption_docs():
    """
    helper function that process adsorption docs, 
    filter out dissociated OOH, 
    and add model0 fingerprint (used in GASpy) to each adsorption doc as one of the keys in doc dictionary
    """
    filters = defaults.adsorption_filters('OOH')
    filters['adsorption_energy'] = {'$gt': -3.5, '$lt': 9.0}
    filters['movement_data.max_adsorbate_movement']['$lt'] = 4.0 #specifically for OOH vs OH

    # Establish the fingerprints that are needed for the preprocessing
    fingerprints = {}
    fingerprints['atoms']='$atoms'
    fingerprints['results']='$results'
    fingerprints['max_surface_movement']='$movement_data.max_slab_movement'
    fingerprints['adsorption_site'] = '$initial_adsorption_site'

    # Pull the documents and then modify them so that they'll work with the preprocessor
    raw_docs = gasdb.get_adsorption_docs(extra_projections=fingerprints, filters=filters)

    from gaspy.mongo import make_atoms_from_doc
    import numpy as np

    def validate_OOH_dissociation(doc):
        """
        helper function that filters out dissociated OOH (O-O bond >1.6A)
        True: OOH not dissciated
        False: OOH dissociated
        """
        if doc['adsorbate']=='OOH':
            atoms = make_atoms_from_doc(doc)
            if np.linalg.norm(atoms.positions[1]-atoms.positions[0])>1.6:
                return False
            else:
                return True
        else:
            return True

    filtered_adsorption_docs = [doc for doc in raw_docs if validate_OOH_dissociation(doc)]

    for doc in filtered_adsorption_docs:
        del doc['atoms']
        del doc['results']
    #generates an N*40 matrix, where N is the number of possible surfaces
    # 40 = 10*4 which means up to 5 elements from each shell in considered, so 10 total, 
    #and each element is represented by a vector of 4 numbers. 
    inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
    outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
    fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter, outer_fingerprinter)
    fps = fingerprinter.fit_transform(filtered_adsorption_docs)
    fingerprints_of_all_docs = [row.reshape(10,4) for row in fps]

    #Add this fingerprint as one of the keys for the doc
    for fp, doc in zip(fingerprints_of_all_docs, filtered_adsorption_docs):
        doc['model0_fingerprint'] = fp
    return filtered_adsorption_docs

import tqdm
import numpy as np
#magpie_python has all the materials properties in periodic table stored 
import magpie
from magpie.data.materials.util.LookUpData import LookUpData
magpie.EffectiveCoordinationNumberAttributeGenerator()


def generate_new_fingerprint(old_fingerprint, atom_num_index, CN_index):
    """
    this helper function takes each fingerprint (a 10X4 list), and flatten it into an array. 
    It does so by taking the atomic number and coordination of each element 
    that are in first and second shell of the surface, and return a vector of 12 numbers, 
    with 11 of them being element properties generated from atomic number, 
    and 12th one is the coordination number of that element

    Args: 

    model0_fingerprint: the model0 fingerprint used in GASpy
    atom_num_index: position of the atomic number in a list (for reshaped model0_fingerprint, it's index1)
    CN_index: position of the coordination number in a list (for reshaped model0_fingerprint, it's index3)

    Return:

    The new fingerprint 

    """
    mass = LookUpData.load_property("AtomicWeight")
    column = LookUpData.load_property("Column")
    row = LookUpData.load_property("Row")
    number = LookUpData.load_property("Number")
    radius = LookUpData.load_property("CovalentRadius")
    en = LookUpData.load_property("Electronegativity")
    s = LookUpData.load_property("NsValence")
    p = LookUpData.load_property("NpValence")
    d = LookUpData.load_property("NdValence")
    f = LookUpData.load_property("NfValence")
    dipole = LookUpData.load_property("DipolePolarizability")

    def fp_to_props(elem_id,num):
        if num != 0: # if there are any elements   
            props = np.array([prop[elem_id-1] for prop in 
                              [mass,column,row,number,radius,en,s,p,d,f,dipole]]+[num])
        else: # if there're no elements, then just place a zero vector with length of 12
            props = np.zeros(12)
        return props


    new_fp = list(np.array([fp_to_props(int(old_fingerprint[i][atom_num_index]),
                                        int(old_fingerprint[i][CN_index])) for i in range(10)]).flatten())+ \
    [np.sum(old_fingerprint[range(5),CN_index]), np.sum(old_fingerprint[range(5,10),CN_index])]  

    return new_fp

def make_fingerprint_from_adsorption_doc(doc):
    """
    this helper function creates an array of numbers for each doc, as its fingerprint
    the fingerprint contains the following information:
    first and second shell of each element that is represented by a vector of 12 numbers. 
    mpid, millerx, millery, millerz, adsorption_site_x, adsorption_site_y, adsorption_site_z, top, shift
    adsorption energy 
    Args: 

    doc (with model) fingerprint

    Return:

    An array of fingerprints with all the final information needed from a doc

    """
    fingerprint = generate_new_fingerprint(doc['model0_fingerprint'],1,3) + \
                                            [doc['mpid'], 
                                            doc['miller'][0], #miller index x coordinate
                                            doc['miller'][1], #miller index y coordinate
                                            doc['miller'][2], #miller index x coordinate
                                            doc['adsorption_site'][0], #adsorption site x coordinate
                                            doc['adsorption_site'][1], #adsorption site y coordinate
                                            doc['adsorption_site'][2], #adsorption site z coordinate
                                            doc['top'], 
                                            doc['shift'],
                                            doc['energy']]
    return fingerprint

import pandas as pd
def make_dataframe(fp_names, 
                   adsorbate1, adsorbate1_adsorption_docs, 
                   adsorbate2, adsorbate2_adsorption_docs, 
                   groupby= None):
    """
    helper function that make a dataframe we can use for ML
    The dataframe contains fhe following information:
    column names
    fingerprints of the local chemistry where the adsorbates are sitting on (an adsorption site or a surface, depends on the "groupby" criteria)
    adsorption enegry of adsorbate1
    adsorption enegry of adsorbate2

    Args: 

    fp_names: column names for dataframe
    adsorbate1, adsorbate2: adsorbates used in scaling relations, in case of ORR, these are OH and OOH
    adsorbate1_adsorption_docs, adsorbate2_adsorption_docs: adsorption docs of adsorbate1, adsorbate2, obtained from GASpy gasdb.get_adsorption_docs
    groupby: grouping criteria, we could choose to group by adsorption site (i.e. obtain the minimum energy of each adsorbate on the same adsorption site of a surface)
    or group by surface (i.e obtain the minimum energy of each adsorbate on the same surface)

    Return:

    An array of fingerprints with all the final information needed from a doc


    """
    #make seperate dataframe of each adsorbate
    df_of_adsorbate1 = pd.DataFrame([make_fingerprint_from_adsorption_doc(doc) for doc in adsorbate1_adsorption_docs],
    columns=fp_names+[adsorbate1])

    df_of_adsorbate2 = pd.DataFrame([make_fingerprint_from_adsorption_doc(doc) for doc in adsorbate2_adsorption_docs],
    columns=fp_names+[adsorbate2])   

    #concatanate the 2 seperate dataframe into a large one                                                    
    df = pd.concat([df_of_adsorbate1,df_of_adsorbate2], axis=0, sort=False)

    #group data based on fingerprints
    if groupby == None:
        grouped_df = df.groupby(fp_names,sort=False,group_keys=False).agg({adsorbate1:'min', adsorbate2:'min'}).dropna()
    else:
        grouped_df = df.groupby(groupby,sort=False,group_keys=False).agg({adsorbate1:'min', adsorbate2:'min'}).dropna()

    grouped_df=grouped_df.reset_index()
    return grouped_df


def make_ORR_dataframe_forML(adsorbate1, adsorbate2):
    """
    helper function that create a dataframe used specifically for ORR scaling relations predictions.

    Args:
    adsorbate1, adsorbate2: in this case OH & OOH

    Returns: 
    grouped_dataframe: a pandas dataframe that contains fingerprints of adsorption sites that have both DFT calculated OH & OOH

    """

    filtered_adsorption_docs = process_adsorption_docs()
    #obtain adsorption_docs of adsorbate1 and 2
    adsorbate1_adsorption_docs = [doc for doc in filtered_adsorption_docs
                                  if doc['adsorbate']== adsorbate1 
                                  and doc['model0_fingerprint'][0][3]>0 
                                  and doc['model0_fingerprint'][5][3]>0]
    adsorbate2_adsorption_docs = [doc for doc in filtered_adsorption_docs
                                  if doc['adsorbate']== adsorbate2 
                                  and doc['model0_fingerprint'][0][3]>0 
                                  and doc['model0_fingerprint'][5][3]>0]

    ### column name for the fingerprints  
    prop_name_temp=['l%dmass%d', # total mass of an element d in shell l 
                    'l%dcolumn%d', # column# (in periodic table) of that element d in shell l 
                    'l%drow%d', # row# (in periodic table) of that element d in shell l 
                    'l%dnumber%d', # atomic number (in periodic table) of that element d in shell l
                    'l%dradius%d', # radius of that element d in shell l
                    'l%den%d', # electro-negativity of that element d in shell l
                    'l%ds%d', # s-orbital of that element d in shell l
                    'l%dp%d', # p-orbital of that element d in shell l
                    'l%dd%d', # d-orbital of that element d in shell l
                    'l%df%d', # f-orbital of that element d in shell l
                    'l%ddipole%d', # dipole of that element d in shell l
                    'l%dCN%d'] # number of atoms of that element d in shell l

    fp_names=list(np.array([[a%(0,i) for a in prop_name_temp] for i in range(5)]).flatten())+ \
             list(np.array([[a%(1,i) for a in prop_name_temp] for i in range(5)]).flatten())+ \
                ['CN0',
                'CN1', 
                'mpid',
                'millerx','millery','millerz', 
                'ads_sitex', 'ads_sitey','ads_sitez', 
                'top', 
                'shift'] 

    #Obtain the DataFrame
    grouped_dataframe = make_dataframe(fp_names, 
                                       adsorbate1, adsorbate1_adsorption_docs, 
                                       adsorbate2, adsorbate2_adsorption_docs, 
                                       groupby= None)
    return grouped_dataframe, fp_names

def make_fingerprints(fp, elems_sets, num):
    """
    helper function that makes fingerprints given a set of elements, 
    fingerprints are atomic number of each element, and # of that element

    Args: 

    elem_set: a set of elements (either first or second shell elements coordinated with an adsorption site)
    num: 0 for first shell, 5 for second shell (this is because in GASpy catalog, we have up to 5 elements for each surfaces)

    """
    i = num
    while i-num <len(set(elems_sets)):
        for element in set(elems_sets):
            atomic_number = getattr(mendeleev, element).atomic_number
            count = elems_sets.count(element)
            fp[i] = [atomic_number,count]
            i = i+1

##outer shell elements, taken fron gaspy_regress.fingerprinters
def concatenate_second_shell(doc):
    """
    helper function that returns what's in the second shell of an adsorption site, taken from GASpy

    """
    second_shell_atoms = []
    for neighbor_coordination in doc['neighborcoord']:
        _, coordination = neighbor_coordination.split(':')
        coordination = coordination.split('-')
        second_shell_atoms.extend(coordination)
    return second_shell_atoms

import mendeleev
def add_fingerprint_to_docs(docs):
    """
    helper function that adds a fingerprint key to each catalog doc dictionary

    """
    for doc in docs:
        fingerprint = np.zeros((10, 2)) #create a 10X2 np.array filled with 0
        binding_atoms = doc['coordination'].split('-')
        #if binding_atoms != ['']: #make sure coordination exists
        second_shell_atoms = concatenate_second_shell(doc) #obtain second shell of elements
        make_fingerprints(fingerprint, binding_atoms,0) #fill in the first shell fingerprint information 
        make_fingerprints(fingerprint, second_shell_atoms, 5) #fill in the second shell fingerprint information
        doc['fingerprint'] = fingerprint #append the fingerprint information into the doc
    return docs


def make_fingerprint_from_catalog_doc(doc):
    """
    this helper function creates an array of numbers for each catalog doc, as its fingerprint
    the fingerprint contains the following information:
    first and second shell of each element is represented by a vector of 12 numbers. 
    with 11 of them being element properties, using fxn:generate_new_fingerprint, and the number of that element
    millerx, millery, millerz, adsorption_site_x, adsorption_site_y, adsorption_site_z, shift
    adsorption energy 
    Args: 

    doc (with model) fingerprint
    atom_num_index: position of the atomic number in a list
    CN_index: position of the coordination number in a list 

    Return:

    An array of fingerprints with all the final information needed from a doc

    """
    fingerprint = generate_new_fingerprint(doc['fingerprint'],0, 1) + \
                                        [doc['miller'][0], #miller index x coordinate
                                        doc['miller'][1], #miller index y coordinate
                                        doc['miller'][2], #miller index x coordinate
                                        doc['adsorption_site'][0], #adsorption site x coordinate
                                        doc['adsorption_site'][1], #adsorption site y coordinate
                                        doc['adsorption_site'][2], #adsorption site z coordinate
                                        doc['shift']] #OH model0 predicted adsorption energy
    return fingerprint


def fit_scaling_relations_residual(adsorbate1, adsorbate2):
    
    """
    helper function that uses TPOTRegressor to train a machine learning model.
    The model uses delta learning Method. The input is thr fingerprint of the adsorption site, 
    and label is the residual of each datapoint deviated from the line. 

    Args:
        scaling_dataframe: the dataframe that has fingerprint of the adsorption sites & adsorption energy of OH & OOH

    Returns:
        hybrid_regressor: the fitted ML pipeline
        ransac: The fitted linear regression model
        X_scaler: the fitted standard scaler

    """
    #####data needed for ML model#####
    grouped_results, fp_names= make_ORR_dataframe_forML(adsorbate1, adsorbate2)
    
    ######fit a linear regression model#####
    ransac = linear_model.RANSACRegressor()

    #convert delta_E to delta_G by adding correction factors for OH and OOH
    X = np.array(grouped_results[[adsorbate1]]-0.223)
    Y = np.array(grouped_results[[adsorbate2]]+0.043)
    ransac.fit(X, Y)
    
    #calculate residuals
    residuals = grouped_results[[adsorbate2]]-ransac.predict(grouped_results[[adsorbate1]])

    ###### hybrid model using scaling relations to predict residual #####
    #prepare dataset (excluded top & mpid info)
    X_scaler = StandardScaler()
    X_hybrid = grouped_results[[fp for fp in fp_names if fp !='mpid' and fp!='top']]
    X_hybrid = X_scaler.fit_transform(X_hybrid)
    y_hybrid = residuals[[adsorbate2]]

    #make, fit, the hybrid tpot pipeline
    #Use MSE to penalize more on the outliers
    tpot_hybrid = TPOTRegressor(generations=2, 
                                population_size=100, 
                                verbosity=2, 
                                scoring = 'neg_mean_squared_error', 
                                n_jobs=16)
    tpot_hybrid.fit(X_hybrid,y_hybrid)
    
    #obtain the best model & do fit overall data into the model
    hybrid_regressor = copy.deepcopy(tpot_hybrid.fitted_pipeline_)
    
    #return the regressor obtained, ransac_regressor (we need the slope & intercept), as well as the fitted scaler
    return hybrid_regressor, ransac, X_scaler