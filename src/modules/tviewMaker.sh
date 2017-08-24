#!/bin/bash


if [ $# -ne 4 ];
then
echo "usage: "$(basename $0) "[bam-file]" "[reference-file]" "[coordinate-start]" "[coordinate-end]"
echo "example: "./$(basename $0) '../../data/NA12878.np.chr3.100kb.0.bam' '../../data/hg19.chr3.9mb.fa' '100000' '200000' '>output.txt'
exit
fi

bam=$1
ref=$2
output=$5
now=$(date '+%d%m%Y%H%M%S')
start=$3
end=$4
step=0
percent=10
samtools index $bam

i=$start
while [ $i -lt $end ]; do
    echo '>'
    samtools tview $bam $ref -d T -p chr3:$i
    ch=$(samtools tview $bam $ref -d T -p chr3:$i | sed -n 2p)
    total=${#ch}
    ch=$(echo $ch | tr -cd "*" | wc -c)
    rest=$(($total-$ch))
    i=$(($i+$rest))
    step=$(($step+1))
    if [ "$step" -ge "10000" ]; then
        >&2 echo "$i bases processed"
        step=0
    fi
done
