#!/bin/bash

sed -n 2,3556522p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr1
sed -n 3556524,7016431p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr2
sed -n 7016433,9849226p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr3
sed -n 9849228,12566578p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr4
sed -n 12566580,15159983p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr5
sed -n 15159985,17600070p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr6
sed -n 17600072,19876443p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr7
sed -n 19876445,21949854p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr8
sed -n 21949856,23926923p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr9
sed -n 23926925,25838316p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr10

sed -n 25838318,27768126p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr11
sed -n 27768128,29672060p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr12
sed -n 29672062,31305838p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr13
sed -n 31305840,32835035p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr14
sed -n 32835037,34292053p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr15
sed -n 34292055,35582602p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr16
sed -n 35582604,36771996p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr17
sed -n 36771998,37920187p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr18
sed -n 37920189,38757583p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr19
sed -n 38757585,39678215p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr20

sed -n 39678217,40345502p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr21
sed -n 40345504,41071482p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chr22
sed -n 41071484,43300639p GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta | awk '{print}' ORS='' > chrX
