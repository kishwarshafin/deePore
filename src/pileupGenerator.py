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

class pileupEncoder:
    '''
    Pileup encoding schema.
    - Lowercase bases represent forward strand mis-match for a
    specific base in the pileup.
    - M represents a match with the reference base.
    - * represents a deletion
    '''
    encoding = {'a': bitarray('10000'), 'A': bitarray('10000'),
                'c': bitarray('01000'), 'C': bitarray('01000'),
                'g': bitarray('00100'), 'G': bitarray('00100'),
                't': bitarray('00010'), 'T': bitarray('00010'),
                'm': bitarray('00000'), 'M': bitarray('00000'),
                '*': bitarray('00001')}
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
                    binaryList = self.getEncodingForBase(pileup_base, ref_base, pileupread.alignment.is_reverse)
                columnList.extend(binaryList)
            for j in range(img.size[1]):
                pixels[i, j] = int(columnList[j]) if j<len(columnList) else 0
            i += 1
        img.save(imgFilename + ".bmp")

    def getEncodingForBase(self, base, ref_base , reverse_flag):
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
        if enChar=='A':
            return [0, 0, 1]
        if enChar=='C':
            return [0, 1, 0]
        if enChar=='G':
            return [0, 1, 1]
        if enChar=='T':
            return [1, 0, 0]
        if enChar=='*':
            return [1, 0, 0]

    def generateBinaryPileup(self, region, baseStart, start, end, sharedArray):
        '''
        Fills a shared memory with bitarray representing the binary pileup.
        :param region: Region in the genome. Ex: chr3
        :param baseStart: Program's actual start site helps to determine the location in shared memory
        :param start: Start site
        :param end: End site
        :param sharedArray: Shared array among processes
        :return:
        '''
        reference = self.refFile.fetch(region, start, end)
        region_bam = "chr" + region
        for pileupcolumn in self.samFile.pileup(region_bam, start, end, truncate=True):
            pBitArray = bitarray()
            ref_base = str(reference[pileupcolumn.pos-start]).upper()
            encodedString = ''
            for pileupread in pileupcolumn.pileups:
                encodedChar = '*'
                if not pileupread.is_del and not pileupread.is_refskip:
                    pileup_base = str(pileupread.alignment.query_sequence[pileupread.query_position]).upper()
                    encodedChar = self.getEncodingForBase(pileup_base, ref_base, pileupread.alignment.is_reverse)
                encodedString += encodedChar
            #print(encodedString)
            pBitArray.encode(pileupEncoder.encoding, encodedString)
            sharedArray[pileupcolumn.pos-baseStart] = pBitArray

    def closeSamFile(self):
        '''
        Closes the samfile.
        :return:
        '''
        self.samFile.close()


def generatePileupDictionary(region, start, end, bamFile, refFile):
    '''
    Returns a pileup dictionary representation.
    Key is the site and value is binary bitarray.
    '''
    cores = multiprocessing.cpu_count()
    processes = []
    currentStart = start
    processManager = Manager()
    sharedDict = processManager.dict()
    for num in range(cores):
        p = pileUpCreator(bamFile, refFile)
        currentEnd = currentStart + int((end-start)/cores) if num < cores-1 else end
        pr = Process(target=p.generateBinaryPileup, args=(region, start, currentStart, currentEnd, sharedDict,))
        pr.start()
        processes.append(pr)
        currentStart = currentEnd

    for p in processes:
        p.join()

    return sharedDict

def generateImageParallel(args):
    '''
    Fill up array from dictionary.
    :param args:
    :return:
    '''
    dictionary, coverage, start = args
    ret = []
    bitStr = dictionary[start].to01()
    for j in range(coverage*6):
        ret.append(not int(bitStr[j]) if j<len(bitStr) else 1)
    return ret

def generateBmpParallel(dictionary, coverage):
    '''
    Uses multiprocessing to fill up the array from bitmap dictionary.
    '''
    pool = Pool(processes=multiprocessing.cpu_count())
    params = [(dictionary, coverage, i) for i in range(len(dictionary))]
    bitMapArray = np.array(pool.map(generateImageParallel, params))
    pool.close()
    return bitMapArray

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

def saveBitmapImage(name, bitMapArray):
    '''
    Save image to file
    '''
    plt.imsave(name, np.array(bitMapArray.T), cmap=cm.gray)

def generateImageLinear(dictionary, coverage, fileName):
    '''
    Linear function to generate the Image.
    Used for testing multiprocessing.
    '''
    img = Image.new('1', (len(dictionary), coverage*6))
    pixels = img.load()
    #print(img.size[0], img.size[1])
    for i in range(img.size[0]):
        bitStr = dictionary[i].to01()
        for j in range(img.size[1]):
            pixels[i, j] = not int(bitStr[j]) if j<len(bitStr) else 1
    img.save(fileName+".bmp")


def generatePileupBasedonVCF(vcf_region, bamFile, refFile, vcfFile, output_dir, window_size):
    vcf_in = VariantFile(vcfFile)
    cnt = 0
    start_timer = timer()
    smry = open(output_dir+'/0.summary'+'-'+vcf_region+".txt", 'w')
    for rec in vcf_in.fetch(vcf_region):
        reg = rec.chrom
        start = rec.pos - window_size - 1
        end = rec.pos + window_size

        filename = output_dir + rec.chrom + "-" + str(rec.pos)
        p = pileUpCreator(bamFile, refFile)
        p.generatePileupLinear(reg, start, end, FLAGS.coverage, filename)
        cnt += 1
        if cnt % 100 == 0:
            end_timer = timer()
            print(str(cnt) + " Records done", file=sys.stderr)
            print("TIME elapsed "+ str(end_timer - start_timer), file=sys.stderr)
        print(rec, end='', file=smry)
        print(start+1, end, filename, file=smry)



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
        "--site_start",
        type=int,
        default = 100000,
        help="Start position in chromosome."
    )
    parser.add_argument(
        "--site_end",
        type=int,
        default = 110000,
        help="End position in chromosome."
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=50,
        help="Read coverage, default is 50x."
    )
    parser.add_argument(
        "--matrix_out",
        type=bool,
        default=False,
        help="If true, will print the matrix in stdout."
    )
    parser.add_argument(
        "--vcf_region_only",
        type=bool,
        default=False,
        help="If true, will generate regions where there are SNPs."
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
    if not FLAGS.vcf_region_only:
        sd = generatePileupDictionary(FLAGS.region, FLAGS.site_start, FLAGS.site_end, FLAGS.bam, FLAGS.ref)
        bitmapArray = generateBmpParallel(sd, FLAGS.coverage)
        saveBitmapImage(FLAGS.output_dir + FLAGS.region + "-" + str(FLAGS.site_start) +".bmp", bitmapArray)

        if FLAGS.matrix_out:
            printBitmapArray(bitmapArray, FLAGS.output_dir + FLAGS.region + "-" + str(FLAGS.site_start) +".txt")
    else:
        generatePileupBasedonVCF(FLAGS.vcf_region, FLAGS.bam, FLAGS.ref, FLAGS.vcf, FLAGS.output_dir, FLAGS.window_size)
