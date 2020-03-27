#!/bin/bash

for k in {1..49..2} {100..3850..250} 7400
do
    echo $k
    python3 auto_tune.py anon_data/ring_mondrian/k${k}_minmaxed.csv class > results/ring_mondrian/test02/k${k}_2hr.txt
done
