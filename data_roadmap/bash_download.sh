#!/bin/bash

## download roadmap data (1032 + 37 + 53 = 1122 tracks)

while read -r line
do
    echo $line
    wget https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/pval/${line} &
    sleep 10s
done < list_histone.txt
wait

while read -r line
do
    echo $line
    wget https://egg2.wustl.edu/roadmap/data/byDataType/dnamethylation/WGBS/FractionalMethylation_bigwig/${line} &
    sleep 3s
done < list_methylation.txt
wait

while read -r line
do
    echo $line
    wget https://egg2.wustl.edu/roadmap/data/byDataType/rna/signal/normalized_bigwig/stranded/${line} &
    sleep 3s
done < list_rna.txt
wait



