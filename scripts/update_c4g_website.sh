#!/bin/sh

# Load GASpy environment and variables
. ~/GASpy/.load_env.sh


#prediction pkl to use
export pklname=CO2RR_predictions_TPOT_FEATURES_coordatoms_chemfp0_neighbors_chemfp0_RESPONSES_energy_BLOCKS_adsorbate

echo $pklname

# Generate the json file to go to elastic search
python convert_prediction_to_json.py ../cache/predictions/$pklname.pkl

# Copy the new data over
scp ../cache/predictions/$pklname.json zulissi@sm1.cheme.cmu.edu:~/CatalystsRE/data/data3.json

# Force the server to restart
ssh -t zulissi@sm1.cheme.cmu.edu "cd CatalystsRE && ./build_run.sh"

# Wait a few seconds then trigger an import
sleep 20
wget -O /dev/null http://sm1.cheme.cmu.edu/import --user=$GASDB_WEB_USER --password=$GASDB_WEB_PW
