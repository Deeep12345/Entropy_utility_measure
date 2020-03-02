#!/bin/bash

for k in {2..12}
do
    echo $k
    python3 auto_tune.py anon_data/ring_mondrian/k${k}_minmaxed.csv class > results/ring_mondrian/k${k}_2hr.txt
done
