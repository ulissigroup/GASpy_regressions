from ase.db import connect
from vasp_settings_to_str import vasp_settings_to_str
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from scipy.sparse import coo_matrix

from matplotlib import rc
rc('text', usetex=False)
mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cPickle as pickle


#Connect to the local adsorption energy database
conAds=connect('../GASpy/adsorption_energy_database.db')
conEnum=connect('../GASpy/enumerated_adsorption_sites.db')

#Only select adsorption energies that have a consistent set of DFT settings
xc='beef-vdw'
encut=350

vasp_settings_str=vasp_settings_to_str({'xc':xc,'encut':encut})

#Dump all the adsorption energies
adsorption_rows=[row for row in conAds.select(**vasp_settings_str) if row.energy<10]
unique_adsorbates=np.unique([row.adsorbate for row in adsorption_rows])

#Grab all of the possible coordination sites
adsorption_rows_catalog=[row for row in conEnum.select()]
unique_coordination_catalog=np.unique([row.coordination for row in adsorption_rows_catalog])
neighborcoord_catalog=[sorted(eval(row.neighborcoord)) for row in adsorption_rows_catalog]
unique_neighborcoord=np.unique([item for sublist in neighborcoord_catalog for item in sublist])

#Make a binarizer so that we can turn the adsorption site labels int
#a vector representing which adsorption type it is
lb=LabelBinarizer()
lb.fit(unique_coordination_catalog)

#Make a binarizer so that we can turn the adsorption site labels int
#a vector representing which adsorption type it is
lb2=LabelBinarizer()
lb2.fit(unique_neighborcoord)


def sumTransform(x):
    neighborarray=np.zeros((len(unique_slab_atoms),len(unique_slab_atoms)))
    for a in x:
        centeratoms=a.split(':')[0]
        coordatoms=a.split(':')[1].split('-')
        for atom in coordatoms:
            if atom!='' and atom!='U':
                neighborarray[unique_slab_atoms.index(centeratoms),unique_slab_atoms.index(atom)]+=1
    return neighborarray.flatten()


unique_slab_atoms=list(np.unique([row for sublist in [row.coordination.split('-') for row in adsorption_rows_catalog] for row in sublist]))

LR_save={}
for adsorbate in unique_adsorbates:

    #Grab rows in the database 
    relevant_rows=[row for row in adsorption_rows if row.adsorbate==adsorbate]
    
    #The adsorption site will be the input to regress on
    adsorption_site=[row.coordination for row in relevant_rows]
    
    #We will try to predict dE
    dE=[row.energy for row in relevant_rows]
    X=lb.transform(adsorption_site)
    
    #Linear regression on site type to get an every energy for each site type:
    LR=LinearRegression()
    LR.fit(X,dE)
    fit_dE=LR.predict(X)
    dE_residual=dE-fit_dE
    
    #Let's try and get the residuals with the coordination of the active site atoms
    secondary_coord=[sorted(eval(row.neighborcoord)) for row in adsorption_rows if row.adsorbate==adsorbate]
    X2=map(sumTransform,secondary_coord)
    
    LR2=LinearRegression()
    LR2.fit(X2,dE_residual)
    fit_dE2=fit_dE+LR2.predict(X2)
    
    #Let's visualize how well this is doing:
    plt.plot(dE,fit_dE2,'.',label='%s, R$^2$=%1.2f, RMSE=%1.2f eV'%(adsorbate,LR2.score(X2,dE_residual),np.sqrt(np.mean((fit_dE2-dE)**2.))))
    
    LR_save[adsorbate]={'primary_LR': LR,'secondary_LR':LR2}
    
#Label the figure and save
plt.xlabel('DFT dE [eV]')
plt.ylabel('Binarized Prediction [eV]')
plt.legend()
plt.plot([-2,5],[-2,5],'--k')
plt.savefig('simple_binarized_regression_secondarycoordination.pdf')

Xenum=lb.transform([row.coordination for row in adsorption_rows_catalog])

secondary_coord=[sorted(eval(row.neighborcoord)) for row in adsorption_rows_catalog]



def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

pool=Pool(16,maxtasksperchild=2)


dEprediction={}
for adsorbate in unique_adsorbates:
    dEprediction[adsorbate]=[]

for chunk in chunks(zip(Xenum,secondary_coord),10000):
    chunk_Xenum,chunk_secondary_coord=zip(*chunk)
    X2enum_chunk=coo_matrix(pool.map(sumTransform,chunk_secondary_coord))
    print('%d/%d'%(len(dEprediction[adsorbate]),len(Xenum)))
    for adsorbate in unique_adsorbates:
        temp=LR_save[adsorbate]['primary_LR'].predict(chunk_Xenum)+LR_save[adsorbate]['secondary_LR'].predict(X2enum_chunk)
        dEprediction[adsorbate]+=list(temp*np.logical_and(temp>-5,temp<5))

pickle.dump(dEprediction,open('primary_coordination_prediction.pkl','w'))

#dEprediction={}
#for adsorbate in unique_adsorbates:
#    dEprediction[adsorbate]=LR_save[adsorbate]['primary_LR'].predict(Xenum) 
#+LR_save[adsorbate]['secondary_LR'].predict(X2enum)
#    dEprediction[adsorbate]=dEprediction[adsorbate]*np.logical_and(dEprediction[adsorbate]>-5,dEprediction[adsorbate]<5)
    
plt.figure()
dE_CO=dEprediction['CO']
dE_H=dEprediction['H']
plt.plot(dE_CO,dE_H,'.')
plt.xlabel('dE CO [eV]')
plt.ylabel('dE H eV]')
plt.savefig('enumerated_CO_H_predictions.pdf')

matching=[]
for dE,row in zip(dEprediction['CO'],adsorption_rows_catalog):
    if dE>-0.9 and dE<-0.5 and 'Al' not in row.formula and 'Cu' not in row.formula:
        matching.append([dE,row])

unique_vals=np.unique(map(lambda x: str([x[0],x[1].formula,x[1].miller,x[1].coordination,x[1].top]),matching))

for match in unique_vals:
    print(match)
    

unique_surfaces=map(eval,np.unique([str([row.mpid,row.miller,row.top,row.shift]) for row in adsorption_rows_catalog]))

minEnergy={}
for mpid,miller,top,shift in unique_surfaces:
    matching_rows=[row[0] for row in enumerate(adsorption_rows_catalog) if row[1].mpid==mpid and row[1].top==top and row[1].shift==shift and row[1].miller==miller]
    minEnergy[str([mpid,miller,top,shift])]={}
    for adsorbate in dEprediction:
        minEnergy[str([mpid,miller,top,shift])][adsorbate]=np.min(dEprediction[adsorbate][matching_rows])

for key in minEnergy:
    if minEnergy[key]['CO']>-0.8 and minEnergy[key]['CO']<-0.2:
        print(key)
        print(minEnergy[key])
#make CO,H binding plot
plt.figure()
dE_CO=map(lambda x: minEnergy[x]['CO'],minEnergy)
dE_H=map(lambda x: minEnergy[x]['H'],minEnergy)
plt.plot(dE_CO,dE_H,'.')
plt.xlabel('dE CO [eV]')
plt.ylabel('dE H eV]')
plt.savefig('enumerated_CO_H_predictions_min.pdf')
    
