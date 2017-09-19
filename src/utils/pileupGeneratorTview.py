import argparse
import pysam
from pysam import VariantFile
from bitarray import bitarray
from sys import getsizeof
import multiprocessing
from multiprocessing import Process, Pool, TimeoutError, Array, Manager
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from timeit import default_timer as timer
import sys
import subprocess
"""
This program takes an alignment file (bam) and a reference file
to create a sparse bitmap representation of the pileup. It uses
pysam's pileup method and encodes each base in pileup to 6 binary
bits. It creates a large binary sparse matrix too.
"""

allSNPS = {}

def getEncoding(ch):
    if not ch:
        return [1, 1, 0]
    elif ch.upper() == ',' or ch.upper()=='.':
        return [0, 0, 0]
    elif ch.upper() == 'A':
        return [0, 0, 1]
    elif ch.upper() == 'C':
        return [0, 1, 0]
    elif ch.upper() == 'G':
        return [0, 1, 1]
    elif ch.upper() == 'T':
        return [1, 0, 0]
    elif ch.upper() == '*':
        return [1, 0, 1]


def tviewToImage(processOutput, start, end, coverage):
    pileupStack = []
    for i in range(coverage+1):
        pileupStack.append([])

    line_no = 0
    baseCount = start-1
    take_upto = 0
    label = []
    labelStr = ""
    while True:
        if processOutput.poll() is not None:
            break
        line = processOutput.stdout.readline().decode('utf-8').rstrip()
        if not line:
            continue

        if line == ">":
            line_no = 0
        elif line_no == 2:
            take_upto = 0
            for ch in line:
                if ch != '*':
                    if baseCount in allSNPS.keys():
                        print("HERE", baseCount)
                    baseCount += 1
                take_upto += 1
                #print(ch, getEncoding(ch))

                label.extend(getEncoding(ch))
                if baseCount == end:
                    break
            labelStr = labelStr + line[:take_upto]
            print(line[:take_upto])
            #print(len(line[:take_upto]))
            #print(label)
            #print(len(label))

        elif line_no != 1 and line_no!=3:
            if line_no-4 > coverage:
                continue
            indx = line_no - 4
            #print(line_no-3)
            print(line[:take_upto])
            for ch in line[:take_upto]:
                pileupStack[indx].extend(getEncoding(ch))
                #print(ch, getEncoding(ch))

        line_no += 1

    pixel2d = []
    pixel2d.append(label)

    for i in range(coverage+1):
        pixel2d.append(pileupStack[i])
    #print(pixel2d)
    img = Image.new('1', (len(pixel2d), len(labelStr)*3))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = pixel2d[i][j]
            #print(pixel2d[i][j],end='')
        #print()
    #img.show()
    #print(len(pileupStack))

def generatePileupBasedonVCF(vcf_region, bamFile, refFile, vcfFile, output_dir, window_size, coverage):
    vcf_in = VariantFile(vcfFile)
    cnt = 0
    start_timer = timer()
    #smry = open(output_dir+'/0.summary'+'-'+vcf_region+".txt", 'w')
    for rec in vcf_in.fetch(vcf_region):
        if len(rec.alleles) == 2:
            ref, al1 = rec.alleles
            if len(ref) == 1 and len(al1) == 1:
                allSNPS[rec.pos] = al1

    for rec in vcf_in.fetch(vcf_region):
        start = rec.pos - window_size - 1
        end = rec.pos + window_size
        if len(rec.alleles) == 2:
            ref, al1 = rec.alleles
            if len(ref) == 1 and len(al1) == 1:
                print(rec.pos, rec.alleles)
                filename = output_dir + rec.chrom + "-" + str(rec.pos)
                rc = subprocess.Popen("./tviewMaker.sh " + bamFile + " " + refFile + " " + str(start) + " " + str(end), shell=True, stdout=subprocess.PIPE)
                tviewToImage(rc, start, end, coverage)
                allSNPS[rec.pos] = al1
                break

        #print(rec, end='', file=smry)
        #print(start+1, end, filename, file=smry)



if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--bam",
        type=str,
        required = True,
        help="BAM file with alignments."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Reference corresponding to the BAM file."
    )
    parser.add_argument(
        "--vcf",
        type=str,
        required=False,
        help="VCF file containing SNPs and SVs."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="3",
        help="Site region. Ex: chr3"
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=50,
        help="Read coverage, default is 50x."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size of pileup."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Name of output directory"
    )

    FLAGS, unparsed = parser.parse_known_args()
    generatePileupBasedonVCF(FLAGS.region, FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir, FLAGS.window_size, FLAGS.coverage)
