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
from modules.model_simple import Inception3
from modules.model import Model
from modules.dataset import PileupDataset, TextColor
import random
import math
import time
import torchnet.meter as meter
np.set_printoptions(threshold=np.nan)


def test(data_file, batch_size, gpu_mode, trained_model, num_classes):
    transformations = transforms.Compose([transforms.ToTensor()])

    validation_data = PileupDataset(data_file, transformations)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=16,
                                   pin_memory=gpu_mode
                                   )
    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = trained_model.eval()
    if gpu_mode:
        model = model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    total_loss = 0
    total_images = 0
    batches_done = 0
    confusion_matrix = meter.ConfusionMeter(num_classes)
    for i, (images, labels, image_name, type_class) in enumerate(validation_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if gpu_mode:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        outputs = model(images)
        confusion_matrix.add(outputs.data.squeeze(), labels.data.type(torch.LongTensor))
        loss = criterion(outputs.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        # Loss count
        total_images += images.size(0)
        total_loss += loss.data[0]

        batches_done += 1
        sys.stderr.write(str(confusion_matrix.conf)+"\n")
        sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(validation_loader)) + "\n" + TextColor.END)

    print('Test Loss: ' + str(total_loss/total_images))
    print('Confusion Matrix: \n', confusion_matrix.conf)

    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(total_loss/total_images) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_base_color(base):
    if base == 'A':
        return 250.0
    if base == 'C':
        return 100.0
    if base == 'G':
        return 180.0
    if base == 'T':
        return 30.0
    if base == '*' or 'N':
        return 5.0


def get_base_by_color(color):
    if color == 250:
        return 'A'
    if color == 100:
        return 'C'
    if color == 180:
        return 'G'
    if color == 30:
        return 'T'
    if color == 5:
        return '*'
    if color == 0:
        return ' '


def get_match_by_color(color):
    if color == 0:
        return ' '
    if color <= 50: #match
        return '.'
    else:
        return 'x' #mismatch

def get_support_by_color(color):
    if color == 254:
        return '.'
    if color == 0:
        return ' '
    if color == 152:
        return 'x'

def test_image(image, img_name):
    image *= 254
    ref_match_channel = image[4,:,:]
    # THIS TESTS THE BASE COLOR CHANNEL
    for column in range(4,ref_match_channel.size(1)):
        str = ""
        for row in range(ref_match_channel.size(0)):
            str += get_base_by_color(math.ceil(image[0,row,column]))
        print(str)

    # THIS TESTS THE MATCH CHANNEL
    # for column in range(4, ref_match_channel.size(1)):
    #     str = ""
    #     for row in range(ref_match_channel.size(0)):
    #         str += get_match_by_color(math.ceil(image[4, row, column]))
    #     print(str)

    # for column in range(4, ref_match_channel.size(1)):
    #     str = ""
    #     for row in range(ref_match_channel.size(0)):
    #         # print(math.ceil(image[5, row, column]))
    #         str += get_support_by_color(math.ceil(image[5, row, column]))
    #     print(str)


def train(train_file, validation_file, batch_size, epoch_limit, file_name, gpu_mode,
          retrain, model_path, only_model, num_classes=3):

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

    # model = Inception3()
    model = Model(inChannel=6, coverageDepth=299, classN=3, leak_value=0.0)
    # model = Model(input_channels=10, depth=28, num_classes=4, widen_factor=8,
    #               drop_rate=0.0, column_width=200, seq_len=seq_len)
    # LOCAL
    # model = Model(input_channels=10, depth=10, num_classes=4, widen_factor=2,
    #               drop_rate=0.0, column_width=300, seq_len=seq_len)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    start_epoch = 0

    if only_model is True:
        model = torch.load(model_path)
        model.train()
        sys.stderr.write(TextColor.PURPLE + 'RETRAIN MODEL LOADED FROM: ' + model_path + "\n" + TextColor.END)
    elif retrain is True:
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        sys.stderr.write(TextColor.PURPLE + 'RETRAIN MODEL LOADED FROM: ' + model_path
                         + ' AT EPOCH: ' + str(start_epoch) + "\n" + TextColor.END)

    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss = 0
        total_images = 0
        start_time = time.time()
        batches_done = 0
        for i, (images, labels, image_name, type) in enumerate(train_loader):
            # print(image_name[0], labels[0])
            # test_image(images[0], image_name)
            # exit()

            print(images.size())
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            x = images
            y = labels

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.contiguous().view(-1, num_classes), y.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            # loss count
            total_images += (x.size(0))
            total_loss += loss.data[0]
            batches_done += 1

            if batches_done % 10 == 0:
                avg_loss = total_loss / total_images if total_images else 0
                print(str(epoch + 1) + "\t" + str(i + 1) + "\t" + str(avg_loss))
                sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Batches done: " + str(batches_done)
                                 + " / " + str(len(train_loader)) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.YELLOW + " Loss: " + str(avg_loss) + "\n" + TextColor.END)
                sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(time.time() - start_time) +
                                 "\n" + TextColor.END)
                start_time = time.time()

        avg_loss = total_loss/total_images if total_images else 0
        sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1)
                         + " Batches done: " + str(i+1) + "/" + str(len(train_loader)) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.YELLOW + " Loss: " + str(avg_loss) + "\n" + TextColor.END)
        print(str(epoch+1) + "\t" + str(i + 1) + "\t" + str(avg_loss))

        if (i+1) % 1000 == 0:
            torch.save(model, file_name + '_checkpoint_' + str(epoch+1) + '_model.pkl')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, file_name + '_checkpoint_' + str(epoch+1) + "." + str(i+1) + "_params.pkl")
            sys.stderr.write(TextColor.RED+" MODEL SAVED \n" + TextColor.END)

        avg_loss = total_loss / total_images if total_images else 0
        sys.stderr.write(TextColor.YELLOW + 'EPOCH: ' + str(epoch+1))
        sys.stderr.write(' Loss: ' + str(avg_loss) + "\n" + TextColor.END)

        torch.save(model, file_name + '_checkpoint_' + str(epoch+1) + '_model.pkl')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, file_name + '_checkpoint_' + str(epoch+1) + "_params.pkl")

        # After each epoch do validation
        test(validation_file, batch_size, gpu_mode, model, num_classes)

    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    torch.save(model, file_name+'_final_model.pkl')
    save_checkpoint({
        'epoch': epoch_limit,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, file_name + '_final_params.pkl')
    sys.stderr.write(TextColor.PURPLE + 'Model saved as:' + file_name + '_final.pkl\n' + TextColor.END)
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
    parser.add_argument(
        "--retrain",
        type=bool,
        default=False,
        help="Pass if want to retrain a previously trained model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to the model to load for retraining purpose"
    )
    parser.add_argument(
        "--only_model",
        type=bool,
        default=False,
        help="Load only model, not the parameters."
    )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.retrain:
        try:
            os.path.isfile(FLAGS.model_path)
        except:
            sys.stderr.write("RETRAIN MODEL FILE DOES NOT EXIST. CHECK model_path PARAMETER.")

    directory_control(FLAGS.model_out.rpartition('/')[0]+"/")
    train(FLAGS.train_file, FLAGS.validation_file, FLAGS.batch_size, FLAGS.epoch_size,
          FLAGS.model_out, FLAGS.gpu_mode, FLAGS.retrain, FLAGS.model_path, FLAGS.only_model)


