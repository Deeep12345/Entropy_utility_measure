#!/bin/bash

for k in {165..200}
do
    cp configs/adult_randoms/datafly${k}.xml config.xml
    echo "${k}  anonymization"
    ./anonymization.sh
    rm config.xml
done
