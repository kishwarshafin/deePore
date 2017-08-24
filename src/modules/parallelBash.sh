#!/bin/bash

if [ $# -ne 5 ];
then
echo "usage: "$(basename $0) "[bam-file]" "[reference-file]" "[coordinate-start]" "[coordinate-end]" "[output-directory]"
echo "example: "./$(basename $0) '../../data/NA12878.np.chr3.100kb.0.bam' '../../data/hg19.chr3.9mb.fa' '100000' '200000' 'output'
exit
fi
. job_pool.sh

bam=$1
ref=$2
output=$5
now=$(date '+%d%m%Y%H%M%S')
start=$3
end=$4
step=0
percent=10
samtools index $bam

tviewDump () {
    bam_l=$1
    ref_l=$2
    start_l=$3
    end_l=$4
    output_l=$5

    i=$start_l
    while [ $i -lt $end_l ]; do
        echo '>'
        samtools tview $bam_l $ref_l -d T -p chr3:$i
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
    return 0
}

CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
bin=$(($end-$start))
bin=$(($bin/$CORES))

i=1
this_start=$start
while [ $i -le $CORES ]; do
    this_end=$(($i*$bin))
    this_end=$(($start+$this_end-1))
    if [ "$i" -eq "$CORES" ];then
        this_end=$end
    fi

    file_name=$this_start"-"$this_end".txt"
    echo $file_name
    tviewDump $bam $ref $this_start $this_end >> $output/$file_name &
    this_start=$(($this_end+1))
    i=$(($i+1))
done
echo "Waiting for processes to finish"
wait
echo "All processes finished!!"

