from pysam import VariantFile
import argparse
import os
from scipy import misc
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


def getDecoded(encoding):
    '''
    Returns binary encoding given a base and it's corresponding
    reference base. The reverse flag is used to determine if the
    match is forward strand match or reverse strand match.
    :param base: Pileup base
    :param ref_base: Reference base
    :param reverse_flag: True if match is in reverse strand
    :return:
    '''
    if encoding == (0, 0, 0):
        return 'M'
    elif encoding == (0, 0, 255):
        return 'A'
    elif encoding == (0, 255, 0):
        return 'C'
    elif encoding == (0, 255, 255):
        return 'G'
    elif encoding == (255, 0, 0):
        return 'T'
    elif encoding == (255, 0, 255):
        return '*'
    else:
        return ' '


def summaryFileReader(summaryFile):
    f = open(summaryFile, "r")
    vcf_rec = ""
    file_path = ""
    i = 1
    for line in f:
        line = line.rstrip()
        if i%2==0:
            print(line)
            file_path = line
            fpList = file_path.split(' ')
            start = fpList[0]
            end = fpList[1]
            imageFile = fpList[2]
            img = Image.open("../"+imageFile+".bmp", 'r')
            #image = misc.imread("../"+imageFile+".bmp", flatten=0)
            pixels = img.load()
            print(img.size[0])
            print(img.size[1])
            for i in range(img.size[0]):
                cnt = 1
                for j in range(img.size[1]):
                    #print(pixels[i,j],end='')
                    if cnt%3==0:
                        fv = (pixels[i,j-2], pixels[i,j-1], pixels[i,j])
                        dChar = getDecoded(fv)
                        print(dChar,end='')
                        #print(cnt)
                    cnt += 1
                print()
            break
        else:
            vcf_rec = line
        i+=1



if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--summary_file",
        type=str,
        required = True,
        help="Summary file name."
    )
    FLAGS, unparsed = parser.parse_known_args()
    summaryFileReader(FLAGS.summary_file)