#!/bin/bash

for k in {107..149}
do
    cp configs/adult_randoms/mondrian${k}.xml config.xml
    echo "${k}  anonymization"
    ./anonymization.sh
    rm config.xml
done
