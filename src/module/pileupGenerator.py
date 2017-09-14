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
"""
This program takes an alignment file (bam) and a reference file
to create a sparse bitmap representation of the pileup. It uses
pysam's pileup method and encodes each base in pileup to 6 binary
bits. It creates a large binary sparse matrix too.
"""
allVariantRecord = {}

class pileUpCreator:
    '''
    Creates pileup from given bam and reference file.
    '''
    def __init__(self, bamFile, fastaFile, vcfFile=""):
        '''
        Set attributes of objects
        :param bamFile: Alignment file
        :param fastaFile: Reference File
        '''
        self.bamFile = bamFile
        self.fastaFile = fastaFile
        self.refFile = pysam.FastaFile(fastaFile)
        self.samFile = pysam.AlignmentFile(bamFile, "rb")
        self.vcfFile = vcfFile

    def generatePileupLinear(self, region, start, end, coverage, imgFilename):
        '''
        Linear program to generate a pileup
        :param region: Region in the genome. Ex: chr3
        :param start: Start site
        :param end: End site
        :return:
        '''
        region_bam = "chr" + region
        img = Image.new('1', (end - start, coverage * 3))
        reference = self.refFile.fetch(region_bam, start, end)
        pixels = img.load()
        i = 0
        for pileupcolumn in self.samFile.pileup(region_bam, start, end, truncate=True):
            columnList = []
            ref_base = str(reference[pileupcolumn.pos - start]).upper()
            for pileupread in pileupcolumn.pileups:
                binaryList = [1, 0, 0]
                if not pileupread.is_del and not pileupread.is_refskip:
                    pileup_base = str(pileupread.alignment.query_sequence[pileupread.query_position]).upper()
                    binaryList = self.getEncodingForBase(pileup_base, ref_base, pileupread.alignment.is_reverse, pileupcolumn.pos)
                columnList.extend(binaryList)
            for j in range(img.size[1]):
                pixels[i, j] = int(columnList[j]) if j<len(columnList) else 0
            i += 1
        img = img.transpose(Image.TRANSPOSE)
        img.save(imgFilename + ".bmp")

    def getEncodingForBase(self, base, ref_base , reverse_flag, pos):
        '''
        Returns binary encoding given a base and it's corresponding
        reference base. The reverse flag is used to determine if the
        match is forward strand match or reverse strand match.
        :param base: Pileup base
        :param ref_base: Reference base
        :param reverse_flag: True if match is in reverse strand
        :return:
        '''
        enChar = base.upper()
        if base==ref_base:
            return [0, 0, 0]
        elif enChar=='A':
            return [0, 0, 1]
        elif enChar=='C':
            return [0, 1, 0]
        elif enChar=='G':
            return [0, 1, 1]
        elif enChar=='T':
            return [1, 0, 0]
        elif enChar=='*':
            return [1, 0, 1]
        else:
            return [1, 1, 0]

    def closeSamFile(self):
        '''
        Closes the samfile.
        :return:
        '''
        self.samFile.close()


def printBitmapArray(bitMapArray, filename):
    '''
    Prints bitMapArray dictionary
    '''
    f = open(filename, 'w')
    for i in range(len(bitMapArray)):
        for j in range(len(bitMapArray[i])):
            print(bitMapArray[i][j],end='', file=f)
        print(file=f)
    f.close()

def getClassForGenotype(gtField):
    if gtField[0] == gtField[-1]:
        return 2 #homozygous alt
    else:
        return 1 #heterozygous

def populateRecordDictionary(vcf_region, vcfFile):
    vcf_in = VariantFile(vcfFile)
    for rec in vcf_in.fetch(vcf_region):
        gtField = str(rec).rstrip().split('\t')[-1].split(':')[0].replace('/', '|').replace('\\', '|').split('|')
        genotypeClass = getClassForGenotype(gtField)
        for i in range(rec.start, rec.stop):
            allVariantRecord[i] = genotypeClass

def getLabel(start, end):
    labelStr = ''
    for i in range(start, end):
        if i in allVariantRecord.keys():
            labelStr += str(allVariantRecord[i])
        else:
            labelStr += str(0)
    return labelStr

def generatePileupBasedonVCF(vcf_region, bamFile, refFile, vcfFile, output_dir, window_size):
    cnt = 0
    start_timer = timer()
    populateRecordDictionary(vcf_region, vcfFile)
    smry = open(output_dir + 'summary' + '-' + vcf_region + ".csv", 'w')
    smry.write('image_file,label\n')
    for rec in VariantFile(vcfFile).fetch(vcf_region):
        reg = rec.chrom
        start = rec.pos - window_size - 1
        end = rec.pos + window_size
        label = getLabel(start, end)
        filename = output_dir + rec.chrom + "-" + str(rec.pos)
        p = pileUpCreator(bamFile, refFile)
        p.generatePileupLinear(reg, start, end, FLAGS.coverage, filename)
        cnt += 1
        if cnt % 1000 == 0:
            end_timer = timer()
            print(str(cnt) + " Records done", file=sys.stderr)
            print("TIME elapsed "+ str(end_timer - start_timer), file=sys.stderr)
        smry.write(filename+".bmp,"+str(label)+'\n')
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
        type = str,
        default = "chr3",
        help="Site region. Ex: chr3"
    )
    parser.add_argument(
        "--vcf_region",
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
        "--output_dir",
        type=str,
        default="output/",
        help="Name of output directory"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size of pileup."
    )
    FLAGS, unparsed = parser.parse_known_args()
    generatePileupBasedonVCF(FLAGS.vcf_region, FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir, FLAGS.window_size)
