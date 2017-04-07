from ase.db import connect
from vasp_settings_to_str import vasp_settings_to_str
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression,BayesianRidge
import matplotlib as mpl
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

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
adsorption_rows=[row for row in conAds.select(**vasp_settings_str) if row.energy<5 and row.energy>-3]
unique_adsorbates=np.unique([row.adsorbate for row in adsorption_rows])

def subtractCentral(nextcoord,coord):
    nextatoms=nextcoord.split('-')
    atoms=coord.split('-')
    for atom in atoms:
        del nextatoms[nextatoms.index(atom)]
    return '-'.join(nextatoms)
    
#Grab all of the possible coordination sites
adsorption_rows_catalog=[row for row in conEnum.select()]
adsorption_rows_catalog=[row for row in conAds.select()]
unique_coordination_catalog=np.unique([row.coordination for row in adsorption_rows_catalog])
unique_nextcoordination_catalog=np.unique([str([row.coordination,subtractCentral(row.nextnearestcoordination,row.coordination)]) for row in adsorption_rows_catalog])

neighborcoord_catalog=[sorted(eval(row.neighborcoord)) for row in adsorption_rows_catalog]
unique_neighborcoord=np.unique([item for sublist in neighborcoord_catalog for item in sublist])

#Make a binarizer so that we can turn the adsorption site labels int
#a vector representing which adsorption type it is
lb=LabelBinarizer()
lb.fit(unique_coordination_catalog)

#Make a binarizer so that we can turn the adsorption site labels int
#a vector representing which adsorption type it is
lb2=LabelBinarizer()
lb2.fit(unique_nextcoordination_catalog)


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
adsorbate_ranges={'CO':[-3,2]}
for adsorbate in unique_adsorbates:
    #Grab rows in the database 
    if adsorbate in adsorbate_ranges:
        relevant_rows=[row for row in adsorption_rows if row.adsorbate==adsorbate and row.energy>adsorbate_ranges[adsorbate][0] and row.energy<adsorbate_ranges[adsorbate][1]]
    else:
        relevant_rows=[row for row in adsorption_rows if row.adsorbate==adsorbate]
    
    #The adsorption site will be the input to regress on
    adsorption_site=[str([row.coordination,subtractCentral(row.nextnearestcoordination,row.coordination)]) for row in relevant_rows]
    
    #We will try to predict dE
    dE=[row.energy for row in relevant_rows]
    X=lb.transform([row.coordination for row in relevant_rows])
    
    X2=lb2.transform(adsorption_site)
    
    X_train, X_test, X2_train,X2_test, y_train, y_test = train_test_split(X,X2,dE, test_size=0.05, random_state=0)
    
    
    #Linear regression on site type to get an every energy for each site type:
    LR=BayesianRidge()
    LR.fit(X_train,y_train)

    LR2=BayesianRidge()
    LR2.fit(X2_train,y_train-LR.predict(X_train))

    
    #Let's visualize how well this is doing:
    plt.plot(y_train,LR.predict(X_train)+LR2.predict(X2_train),'.',label='%s, R$^2$=%1.2f, RMSE=%1.2f eV'%(adsorbate,LR.score(X_train,y_train),np.sqrt(np.mean((LR.predict(X_train)+LR2.predict(X2_train)-y_train)**2.))))
    
    plt.plot(y_test,LR.predict(X_test)+LR2.predict(X2_test),'.',label='%s, R$^2$=%1.2f, RMSE=%1.2f eV'%(adsorbate,LR.score(X_test,y_test),np.sqrt(np.mean((LR.predict(X_test)+LR2.predict(X2_test)-y_test)**2.))))

    
    LR_save[adsorbate]={'primary_LR': LR,'secondary_LR':LR2}
    
#Label the figure and save
plt.xlabel('DFT dE [eV]')
plt.ylabel('Binarized Prediction [eV]')
plt.legend(loc=4)
plt.plot([-2,5],[-2,5],'--k')
plt.savefig('simple_binarized_regression_next.pdf')

Xenum=lb.transform([row.coordination for row in adsorption_rows_catalog])

secondary_coord=[str([row.coordination,subtractCentral(row.nextnearestcoordination,row.coordination)]) for row in adsorption_rows_catalog]

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
    X2enum_chunk=lb2.transform(chunk_secondary_coord)
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
    
