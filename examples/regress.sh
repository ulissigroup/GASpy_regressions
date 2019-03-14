#!/bin/sh
shifter bash <<EOF
python -c 'from gaspy_regress import fit_model0_adsorption_energies; fit_model0_adsorption_energies()'
df
hostname
EOF
exit
