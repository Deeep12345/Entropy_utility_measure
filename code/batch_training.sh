#!/bin/bash

for k in {2..10}
do
    echo $k
    python3 auto_tune.py anon_data/datafly/datafly${k}_cat.csv salary > results/datafly${k}_12hr.txt
done
