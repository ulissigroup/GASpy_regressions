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


#Connect to the local adsorption energy database
conAds=connect('../manage_DFT_database/adsorption_energy_database.db')

#Only select adsorption energies that have a consistent set of DFT settings
xc='beef-vdw'
encut=350

vasp_settings_str=vasp_settings_to_str({'xc':xc,'encut':encut})

#Dump all the adsorption energies
adsorption_rows=[row for row in conAds.select(**vasp_settings_str) if row.energy<10]

unique_adsorbates=np.unique([row.adsorbate for row in adsorption_rows])

unique_elements=np.unique([item for sublist in [row.symbols for row in adsorption_rows] for item in sublist])
lbElement=LabelBinarizer()
lbElement.fit(unique_elements)

    
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
    
    plt.plot(dE,fit_dE,'.',label='%s, R$^2$=%1.2f, RMSE=%1.2f eV'%(adsorbate,LR.score(X,dE),np.sqrt(np.mean((fit_dE-dE)**2.))))
    
#Label the figure and save
plt.xlim([-3,5])
plt.ylim([-3,5])
plt.xlabel('DFT dE [eV]')
plt.ylabel('Binarized Prediction [eV]')
plt.legend(loc='lower right')
plt.plot([-2,5],[-2,5],'--k')
plt.savefig('simple_binarized_count.pdf')
