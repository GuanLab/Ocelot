#!/bin/bash

mkdir -p bedgraph

# bigwig to bedgraph
for file in ./bigwig/*
do
    #echo $file # ./bigwig/C51M22.bigwig
    name=${file##*/}
    #echo $name # C51M22.bigwig
    IFS='.' read -ra ADDR <<< "$name"
    #echo ${ADDR[0]} # C51M22
    ./bigWigToBedGraph $file ./bedgraph/${ADDR[0]}.txt
done

# orange feature
for file in ./bedgraph/*txt
do
    cell=${file##*/}
    echo $cell
    time perl produce_frequency.pl $cell ./bedgraph/ ./orange/ ./submission_template/submission_template.bedgraph
done

# 4min per id
time perl produce_frequency_rank.pl ./orange/ ./orange_rank/ 



