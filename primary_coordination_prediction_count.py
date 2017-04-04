from ase.db import connect
from vasp_settings_to_str import vasp_settings_to_str
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

from matplotlib import rc
rc('text', usetex=False)
mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool


#Connect to the local adsorption energy database
conAds=connect('../manage_DFT_database/adsorption_energy_database.db')
conEnum=connect('../manage_DFT_database/enumerated_adsorption_sites.db')

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

unique_elements=np.unique([item for sublist in [row.symbols for row in adsorption_rows] for item in sublist])
lbElement=LabelBinarizer()
lbElement.fit(unique_elements)

LR_save={}
for adsorbate in unique_adsorbates:
#Grab rows in the database 
    relevant_rows=[row for row in adsorption_rows if row.adsorbate==adsorbate]
    
    #The adsorption site will be the input to regress on
    adsorption_site=[row.coordination for row in relevant_rows]
    
    #We will try to predict dE
    dE=[row.energy for row in relevant_rows]
    
    #Make a binarizer so that we can turn the adsorption site labels int
    #a vector representing which adsorption type it is
    X=np.array(map(lambda y: np.sum(lbElement.transform(y.split('-')),axis=0),[row.coordination for row in relevant_rows]))
    
    #Linear regression on site type to get an every energy for each site type:
    LR=LinearRegression()
    LR.fit(X,dE)
    fit_dE=LR.predict(X)
    dE_residual=dE-fit_dE
    

    plt.plot(dE,fit_dE,'.',label='%s, R$^2$=%1.2f, RMSE=%1.2f eV'%(adsorbate,LR.score(X,dE),np.sqrt(np.mean((fit_dE-dE)**2.))))
    LR_save[adsorbate]={'primary_LR': LR}
    
#Label the figure and save
plt.xlabel('DFT dE [eV]')
plt.ylabel('Binarized Prediction [eV]')
plt.legend()
plt.plot([-2,5],[-2,5],'--k')
plt.savefig('simple_binarized_regression_count.pdf')


Xenum=np.array(map(lambda y: np.sum(lbElement.transform(y.split('-')),axis=0),[row.coordination for row in adsorption_rows_catalog]))



#pool=Pool(8)
#X2enum=pool.map(sumTransform,secondary_coord)
    
dEprediction={}
for adsorbate in unique_adsorbates:
    dEprediction[adsorbate]=LR_save[adsorbate]['primary_LR'].predict(Xenum) #+LR_save[adsorbate]['secondary_LR'].predict(X2enum)
    dEprediction[adsorbate]=dEprediction[adsorbate]*np.logical_and(dEprediction[adsorbate]>-5,dEprediction[adsorbate]<5)
    
plt.figure()
dE_CO=dEprediction['CO']
dE_H=dEprediction['H']
plt.plot(dE_CO,dE_H,'.')
plt.xlabel('dE CO [eV]')
plt.ylabel('dE H eV]')
plt.savefig('enumerated_CO_H_predictions_count.pdf')
    
    

unique_surfaces=map(eval,np.unique([str([row.mpid,row.miller,row.top,row.shift]) for row in adsorption_rows_catalog]))

minEnergy={}
for mpid,miller,top,shift in unique_surfaces:
    matching_rows=[row[0] for row in enumerate(adsorption_rows_catalog) if row[1].mpid==mpid and row[1].top==top and row[1].shift==shift and row[1].miller==miller]
    key=str([adsorption_rows_catalog[matching_rows[0]].formula,mpid,miller,top,shift])
    minEnergy[key]={}
    for adsorbate in dEprediction:
        minEnergy[key][adsorbate]=np.min(dEprediction[adsorbate][matching_rows])

for key in sorted(minEnergy,key=lambda x: minEnergy[x]['CO']):
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
plt.savefig('enumerated_CO_H_predictions_min_count.pdf')
    
