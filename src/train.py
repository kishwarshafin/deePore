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
from modules.model import Model
from modules.dataset import PileupDataset, TextColor
import random
np.set_printoptions(threshold=np.nan)


def test(data_file, batch_size, gpu_mode, trained_model,seq_len):
    transformations = transforms.Compose([transforms.ToTensor()])

    validation_data = PileupDataset(data_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=16,
                                   pin_memory=gpu_mode
                                   )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = trained_model.eval()
    if gpu_mode:
        model = model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    for i, (images, labels) in enumerate(validation_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if gpu_mode:
            images = images.cuda()
            labels = labels.cuda()

        for row in range(images.size(2)):
            # segmentation of image. Currently using 1xCoverage
            left = max(0, row - seq_len)
            right = min(row + seq_len, images.size(2))
            x = images[:, :, left:right, :]
            y = labels[:, row]

            # Pad the images if window size doesn't fit
            if row - left < seq_len:
                padding = Variable(torch.zeros(x.size(0), x.size(1), seq_len - (row - left), x.size(3)))
                if gpu_mode:
                    padding = padding.cuda()
                x = torch.cat((padding, x), 2)
            if right - row < seq_len:
                padding = Variable(torch.zeros(x.size(0), x.size(1), seq_len - (right - row), x.size(3)))
                if gpu_mode:
                    padding = padding.cuda()
                x = torch.cat((x, padding), 2)

            total_variation = int(torch.sum(y.eq(0)).data[0] / 2)
            total_variation += torch.sum(y.eq(2)).data[0]
            total_variation += torch.sum(y.eq(3)).data[0]

            if total_variation == 0 and random.uniform(0, 1) * 100 > 5:
                continue
            elif random.uniform(0, 1) < total_variation / batch_size < 0.02:
                continue

            # Forward + Backward + Optimize
            outputs = model(x)
            loss = criterion(outputs, y)

            # loss count
            total_images += batch_size
            total_loss += loss.data[0]

    print('Test Loss: ' + str(total_loss/total_images))
    sys.stderr.write('Test Loss: ' + str(total_loss/total_images) + "\n")


def get_window(index, window_size, length):
    if index - window_size < 0:
        return 0, index + window_size + (window_size-index)
    elif index + window_size >= length:
        return index - window_size - (window_size - (length - index)), length
    return index - window_size, index + window_size


def train(train_file, validation_file, batch_size, epoch_limit, file_name, gpu_mode, seq_len):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_data_set = PileupDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=gpu_mode
                              )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = Model(input_channels=4, depth=28, num_classes=4, widen_factor=8,
                  drop_rate=0.0, column_width=200, seq_len=seq_len*2)

    #LOCAL
    # model = Model(input_channels=4, depth=10, num_classes=4, widen_factor=2,
    #               drop_rate=0.0, column_width=200, seq_len=seq_len * 2)
    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(epoch_limit):
        total_loss = 0
        total_images = 0
        for i, (images, labels) in enumerate(train_loader):

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
                left = max(0, row-seq_len)
                right = min(row+seq_len, images.size(2))
                x = images[:, :, left:right, :]
                y = labels[:, row]

                # Pad the images if window size doesn't fit
                if row-left < seq_len:
                    padding = Variable(torch.zeros(x.size(0), x.size(1), seq_len-(row-left), x.size(3)))
                    if gpu_mode:
                        padding = padding.cuda()
                    x = torch.cat((padding, x), 2)
                if right-row < seq_len:
                    padding = Variable(torch.zeros(x.size(0), x.size(1), seq_len - (right - row), x.size(3)))
                    if gpu_mode:
                        padding = padding.cuda()
                    x = torch.cat((x, padding), 2)

                total_variation = int(torch.sum(y.eq(0)).data[0] / 2)
                total_variation += torch.sum(y.eq(2)).data[0]
                total_variation += torch.sum(y.eq(3)).data[0]

                if total_variation == 0 and random.uniform(0, 1)*100 > 5:
                    continue
                elif random.uniform(0, 1) < total_variation/batch_size < 0.02:
                    continue

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(x)

                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # loss count
                total_images += batch_size
                total_loss += loss.data[0]

            avg_loss = total_loss/total_images if total_images else 0
            sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch) + " Batches done: " + str(i+1))
            sys.stderr.write(" Loss: " + str(avg_loss) + "\n" + TextColor.END)
            print(str(epoch) + "\t" + str(i + 1) + "\t" + str(avg_loss))

        # After each epoch do validation
        avg_loss = total_loss / total_images if total_images else 0
        test(validation_file, batch_size, gpu_mode, model, seq_len)
        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch))
        sys.stderr.write(' Loss: ' + str(avg_loss) + "\n" + TextColor.END)

        torch.save(model, file_name + '_checkpoint_' + str(epoch) + '.pkl')
        torch.save(model.state_dict(), file_name + '_checkpoint_' + str(epoch) + '_params' + '.pkl')

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    torch.save(model, file_name+'_final.pkl')
    sys.stderr.write(TextColor.PURPLE + 'Model saved as:' + file_name + '_final.pkl\n' + TextColor.END)

    torch.save(model.state_dict(), file_name+'_final_params'+'.pkl')
    sys.stderr.write(TextColor.PURPLE + 'Model parameters saved as:' + file_name + '_final_params.pkl\n' + TextColor.END)


def directory_control(file_path):
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
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
        "--seq_len",
        type=int,
        required=False,
        default=10,
        help="Sequence to look at while doing prediction."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=False,
        default='./model',
        help="Path and file_name to save model, default is ./model"
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )

    FLAGS, unparsed = parser.parse_known_args()
    directory_control(FLAGS.model_out.rpartition('/')[0]+"/")
    train(FLAGS.train_file, FLAGS.validation_file, FLAGS.batch_size, FLAGS.epoch_size,
          FLAGS.model_out, FLAGS.gpu_mode, FLAGS.seq_len)


