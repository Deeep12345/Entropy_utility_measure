#!/bin/bash

for k in {1..200}
do
    cp configs/birth_randoms/datafly${k}_shuffled.xml config.xml
    echo "${k}  anonymization"
    ./anonymization.sh
    rm config.xml
done
