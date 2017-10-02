import argparse
import os
import sys
from PIL import Image, ImageOps
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.model import CNN
from modules.dataset import PileupDataset, TextColor


def train(csvFile, batchSize, epochLimit, fileName, gpu_mode):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    trainDset = PileupDataset(csvFile, transformations)
    trainLoader = DataLoader(trainDset,
                             batch_size=batchSize,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=gpu_mode # CUDA only
                             )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    cnn = CNN()
    if gpu_mode:
        cnn = cnn.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(epochLimit):
        total_loss = 0
        total_images = 0
        for i, (images, labels) in enumerate(trainLoader):
            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            for row in range(images.size(2)):
                # segmentation of image. Currently using 1xCoverage
                x = images[:, :, row:row + 1, :]
                y = labels[:, row]

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = cnn(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # loss count
                total_images += batchSize
                total_loss += loss.data[0]

            # if (i+1) % 1 == 0:
            sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch) + " Batches done: " + str(i+1))
            sys.stderr.write(" Loss: " + str(total_loss/total_images) + "\n" + TextColor.END)

        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch))
        sys.stderr.write(' Images: ' + str(total_images) + ' Loss: ' + str(total_loss/total_images) + "\n" + TextColor.END)

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    torch.save(cnn, fileName+'.pkl')

    sys.stderr.write(TextColor.PURPLE + 'Model saved as:' + fileName + '.pkl\n' + TextColor.END)

    torch.save(cnn.state_dict(), fileName+'-params'+'.pkl')

    sys.stderr.write(TextColor.PURPLE + 'Model parameters saved as:' + fileName + '-params.pkl\n' + TextColor.END)

if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for training, default is 100."
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        required=False,
        default=10,
        help="Epoch size for training iteration."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=False,
        default='./CNN',
        help="Path and filename to save model, default is ./CNN"
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    FLAGS, unparsed = parser.parse_known_args()

    train(FLAGS.csv_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode)


