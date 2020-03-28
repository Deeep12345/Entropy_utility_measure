#!/bin/bash

for k in {2..20..2} {35..725..15} 1887
do
    mv configs/mondrian${k}.xml config.xml
    echo "${k} anonymization"
    ./anonymization.sh
    mv config.xml configs/mondrian${k}.xml
done
