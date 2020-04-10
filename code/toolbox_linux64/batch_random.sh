#!/bin/bash

for k in {1..200}
do
    cp configs/mondrian${k}.xml config.xml
    echo "${k} anonymization"
    ./anonymization.sh
    rm config.xml
done
