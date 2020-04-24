#!/bin/bash

for k in {1..200}
do
for shuff in "_shuffled" ""
do
    cp configs/ring_randoms/datafly${k}${shuff}.xml config.xml
    echo "${k} ${shuff} anonymization"
    ./anonymization.sh
    rm config.xml
done
done
