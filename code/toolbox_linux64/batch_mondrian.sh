#!/bin/bash

for k in {2..12}
do
    mv configs/mondrian${k}.xml config.xml
    ./anonymization.sh
    echo "${k} anonymization"
    mv config.xml configs/mondrian${k}.xml
done
