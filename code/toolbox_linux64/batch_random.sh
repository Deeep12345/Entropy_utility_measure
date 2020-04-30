#!/bin/bash

for k in {1..200}
do
    cp configs/ring_randoms/mondrian${k}.xml config.xml
    echo "${k} ${shuff} anonymization"
    ./anonymization.sh
    rm config.xml
done
