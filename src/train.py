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
from torch.nn.utils.rnn import pack_padded_sequence

def train(csvFile, batchSize, epochLimit, fileName):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    trainDset = PileupDataset(csvFile, transformations)
    trainLoader = DataLoader(trainDset,
                             batch_size=batchSize,
                             shuffle=False,
                             num_workers=4
                             # pin_memory=True # CUDA only
                             )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    cnn = CNN()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    #print('params', list(cnn.parameters()))
    optimizer = torch.optim.Adam(cnn.parameters())

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(epochLimit):
        total_loss = 0
        total_images = 0
        for i, (images, labels) in enumerate(trainLoader):
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            length = outputs.size(2)
            loss = criterion(outputs.view(-1, length), labels.view(-1))

            #loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            #print(loss)
            optimizer.step()
            #print(loss)

            # loss count
            total_images += batchSize
            total_loss += loss

            #if (i+1) % 100 == 0:
            sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch) + " Batches done: " + str(i+1))
            sys.stderr.write("Loss: " + str(total_loss.data[0]/total_images) + "\n" + TextColor.END)

        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch))
        sys.stderr.write(' Images: ' + str(total_images) + ' Loss: ' + str(total_loss.data[0]/total_images) + "\n" + TextColor.END)

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
    FLAGS, unparsed = parser.parse_known_args()

    train(FLAGS.csv_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out)


