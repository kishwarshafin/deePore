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
import random


def validate(csvFile, batchSize, gpu_mode, trained_model):
    transformations = transforms.Compose([transforms.ToTensor()])

    validation_data = PileupDataset(csvFile, transformations)
    validation_loader = DataLoader(validation_data,
                             batch_size=batchSize,
                             shuffle=True,
                             num_workers=16,
                             pin_memory=gpu_mode # CUDA only
                             )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = trained_model.eval()
    if gpu_mode:
        model = model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    for i, (images, labels) in enumerate(validation_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

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
            outputs = model(x)
            loss = criterion(outputs, y)

            # loss count
            total_images += batchSize
            total_loss += loss.data[0]

    print('Validation Loss: ' + str(total_loss/total_images))
    sys.stderr.write('Validation Loss: ' + str(total_loss/total_images) + "\n")


def get_window(index, window_size, length):
    if index - window_size < 0:
        return 0, index + window_size + (window_size-index)
    elif index + window_size >= length:
        return index - window_size - (window_size - (length - index)), length
    return index - window_size, index + window_size


def train(train_file, validation_file, batchSize, epochLimit, fileName, gpu_mode):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    trainDset = PileupDataset(train_file, transformations)
    trainLoader = DataLoader(trainDset,
                             batch_size=batchSize,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=gpu_mode
                             )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    cnn = CNN()
    if gpu_mode:
        cnn = torch.nn.DataParallel(cnn).cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(epochLimit):
        total_loss = 0
        total_images = 0
        total_could_be = 0
        for i, (images, labels) in enumerate(trainLoader):

            # if batch size not distributable among all GPUs then skip
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            for row in range(images.size(2)):
                # segmentation of image. Currently using 1xCoverage
                # (l, r) = get_window(row, 5, images.size(2))
                x = images[:, :, row:row+1, :]
                y = labels[:, row]
                sum = torch.sum(y).data[0]
                total_could_be += batchSize

                if sum == 0 and random.uniform(0, 1)*100 > 5:
                    continue
                elif sum/batchSize < 0.02 and sum/batchSize > random.uniform(0, 1):
                    continue

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = cnn(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # loss count
                total_images += batchSize
                total_loss += loss.data[0]


            sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch) + " Batches done: " + str(i+1))
            sys.stderr.write(" Loss: " + str(total_loss/total_images) + "\n" + TextColor.END)
            print(str(epoch) + "\t" + str(i + 1) + "\t" + str(total_loss/total_images))

        # After each epoch do validation
        validate(validation_file, batchSize, gpu_mode, cnn)
        sys.stderr.write(TextColor.YELLOW + 'Could be: ' + str(total_could_be) + ' Chosen: ' + str(total_images) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch))
        sys.stderr.write(' Loss: ' + str(total_loss/total_images) + "\n" + TextColor.END)
        torch.save(cnn, fileName + '_checkpoint_' + str(epoch) + '.pkl')
        torch.save(cnn.state_dict(), fileName + '_checkpoint_' + str(epoch) + '-params' + '.pkl')

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)
    torch.save(cnn, fileName+'_final.pkl')

    sys.stderr.write(TextColor.PURPLE + 'Model saved as:' + fileName + '.pkl\n' + TextColor.END)
    torch.save(cnn.state_dict(), fileName+'_final_params'+'.pkl')

    sys.stderr.write(TextColor.PURPLE + 'Model parameters saved as:' + fileName + '-params.pkl\n' + TextColor.END)

if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training data description csv file."
    )
    parser.add_argument(
        "--validation_file",
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

    train(FLAGS.train_file, FLAGS.validation_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode)


