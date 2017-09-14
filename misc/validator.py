import os
import argparse
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from timeit import default_timer as timer
import sys
import torchvision
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

def summaryFileReader(csvFile):
    dataFrame = pd.read_csv(csvFile)
    for index, row in dataFrame.iterrows():
        #print(row['image_file'], row['label'])
        img = Image.open(row['image_file'], 'r')
        pixels = img.load()
        for i in range(img.size[0]):
            cnt = 1
            for j in range(img.size[1]):
                # print(pixels[i,j],end='')
                if cnt % 3 == 0:
                    fv = (pixels[i, j - 2], pixels[i, j - 1], pixels[i, j])
                    dChar = getDecoded(fv)
                    print(dChar, end='')
                    # print(cnt)
                cnt += 1
            print()
        break
    #print(dataFrame['image_file'])




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