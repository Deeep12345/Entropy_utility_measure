#!/bin/bash

for k in {1..200}
do
    cp configs/heart_randoms/datafly${k}.xml config.xml
    echo "${k}  datafly anonymization"
    ./anonymization.sh
    rm config.xml
    cp configs/heart_randoms/datafly${k}_shuffled.xml config.xml
    echo "${k}  datafly shuffled  anonymization"
    ./anonymization.sh
    rm config.xml
done
