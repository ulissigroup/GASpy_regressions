#!/bin/sh
shifter bash <<EOF
python -c 'from gaspy_regress import cache_predictions; cache_predictions()'
df
hostname
EOF
exit
