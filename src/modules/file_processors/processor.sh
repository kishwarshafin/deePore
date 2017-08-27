#!/bin/bash
if [ $# -ne 1 ];
then
echo "usage: "$(basename $0) "[vcf-file]"
echo "example: "./$(basename $0) '../data/HG001.GRCh37.chr3.100kb.vcf.gz'
exit
fi

vcf_file=$1
echo $vcf_file

chrs=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y MT)
for i in ${chrs[@]};
    do
        python vcfParser.py --vcf $vcf_file --region $i
    done
