#!/bin/bash

for k in {2..10}
do
    mv configs/incognito${k}.xml config.xml
    ./anonymization.sh
    echo "${k} anonymization"
    mv config.xml configs/incognito${k}.xml 
done  
