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
from modules.model_simple import Model
from modules.dataset import PileupDataset, TextColor
import random
import math
import time
import torchnet.meter as meter
np.set_printoptions(threshold=np.nan)


def test(data_file, batch_size, gpu_mode, trained_model, seq_len, num_classes):
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
    for i, (images, labels, image_name) in enumerate(validation_loader):
        if gpu_mode is True and images.size(0) % 8 != 0:
            continue

        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        if gpu_mode:
            images = images.cuda()
            labels = labels.cuda()

        for row in range(images.size(2)):
            # segmentation of image. Currently using 1xCoverage
            # segmentation of image. Currently using 1xCoverage
            x = images[:, :, row:row+seq_len, :]
            y = labels[:, row:row+seq_len]

            total_variation = int(torch.sum(y.eq(0)).data[0] / 2)
            total_variation += torch.sum(y.eq(2)).data[0]
            total_variation += torch.sum(y.eq(3)).data[0]
            chance = random.uniform(0, 1)

            if total_variation == 0 and chance > 0.05:
                continue
            elif chance < 0.02 and total_variation / (batch_size * seq_len) <= 0.02:
                continue

            # Forward + Backward + Optimize
            outputs = model(x)

            # for each_base in range(seq_len):
            #     confusion_matrix.add(outputs[:, each_base, :].data.squeeze(), y[:, each_base].data.type(torch.LongTensor))
            confusion_matrix.add(outputs.data.squeeze(), y.data.type(torch.LongTensor))
            loss = criterion(outputs.contiguous().view(-1, num_classes), y.contiguous().view(-1))

            # Loss count
            total_images += (batch_size * seq_len)
            total_loss += loss.data[0]
        batches_done += 1
        print(confusion_matrix.conf)
        sys.stderr.write(TextColor.BLUE+'Batches done: ' + str(batches_done) + " / " + str(len(validation_loader)) + "\n" + TextColor.END)


    print('Test Loss: ' + str(total_loss/total_images))
    sys.stderr.write(TextColor.YELLOW+'Test Loss: ' + str(total_loss/total_images) + "\n"+TextColor.END)
    sys.stderr.write("Confusion Matrix \n: " + str(confusion_matrix.conf) + "\n" + TextColor.END)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def train(train_file, validation_file, batch_size, epoch_limit, file_name, gpu_mode,
          seq_len, retrain, model_path, only_model, num_classes=4):

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

    model = Model(inChannel=10, coverageDepth=200, classN=4, window_size=1, leak_value=0.0)
    # model = CNN(inChannel=10, outChannel=250, coverageDepth=300, classN=4, window_size=1)
    # model = Model(input_channels=10, depth=28, num_classes=4, widen_factor=8,
    #               drop_rate=0.0, column_width=200, seq_len=seq_len)
    # LOCAL
    # model = Model(input_channels=10, depth=10, num_classes=4, widen_factor=2,
    #               drop_rate=0.0, column_width=300, seq_len=seq_len)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
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

        for i, (images, labels, image_name) in enumerate(train_loader):

            # if batch size not distributable among all GPUs then skip
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue
            start_time = time.time()

            images = Variable(images)
            labels = Variable(labels)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()
            jump_iterator = random.randint(1, int(math.ceil(seq_len/2)))
            for row in range(0, images.size(2), jump_iterator):
                jump_iterator = random.randint(1, int(math.ceil(seq_len / 2)))
                # to_index = min(row+seq_len, images.size(2) - 1)
                # segmentation of image. Currently using 1xCoverage
                x = images[:, :, row:row+seq_len, :]
                y = labels[:, row:row+seq_len]

                total_variation = int(torch.sum(y.eq(0)).data[0]/2)
                total_variation += torch.sum(y.eq(2)).data[0]
                total_variation += torch.sum(y.eq(3)).data[0]
                chance = random.uniform(0, 1)

                if total_variation == 0 and chance > 0.05:
                    continue
                elif chance < 0.02 and total_variation/(batch_size*seq_len) <= 0.02:
                    continue
                # print("Selected: ", total_variation/(batch_size*seq_len), chance)
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs.contiguous().view(-1, num_classes), y.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                # loss count
                total_images += (batch_size*seq_len)
                total_loss += loss.data[0]

            avg_loss = total_loss/total_images if total_images else 0
            end_time = time.time()
            sys.stderr.write(TextColor.BLUE + "EPOCH: " + str(epoch+1) + " Batches done: " + str(i+1) + "/" + str(len(train_loader)))
            sys.stderr.write(" Loss: " + str(avg_loss) + "\n" + TextColor.END)
            sys.stderr.write(TextColor.DARKCYAN + "Time Elapsed: " + str(end_time-start_time) + "\n" + TextColor.END)
            print(str(epoch+1) + "\t" + str(i + 1) + "\t" + str(avg_loss))
            if (i+1) % 100 == 0:
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
        test(validation_file, batch_size, gpu_mode, model, seq_len, num_classes)

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
        "--seq_len",
        type=int,
        required=False,
        default=1,
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
          FLAGS.model_out, FLAGS.gpu_mode, FLAGS.seq_len, FLAGS.retrain, FLAGS.model_path, FLAGS.only_model)


