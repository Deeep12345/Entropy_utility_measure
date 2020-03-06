#!/bin/bash

for k in {100..7350..250}
do
    mv configs/mondrian${k}.xml config.xml
    ./anonymization.sh
    echo "${k} anonymization"
    mv config.xml configs/mondrian${k}.xml
done
