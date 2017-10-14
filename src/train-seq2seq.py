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
from modules.model import EncoderRNN, AttnDecoderRNN
from modules.dataset import PileupDataset, TextColor
from torch.nn.utils.rnn import pack_padded_sequence

n_layers = 5
hidden_size = 500

coverage = 34
classN = 5
window_size = 51
SOS_TOKEN = 4
EOS_token = 5


def evaluate(train_file, batch_size, gpu_mode, encoder, decoder, max_length=51):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    test_dataset = PileupDataset(train_file, transformations)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=gpu_mode  # CUDA only
                             )
    confusion_meter = np.zeros((5, 5), dtype=np.int)
    for i, (images, labels) in enumerate(test_loader):
        images = Variable(images).cuda() if gpu_mode else Variable(images)
        #labels = Variable(labels).cuda() if gpu_mode else Variable(labels)

        encoder_hidden = encoder.initHidden(images.size(0))
        encoder_outputs = Variable(torch.zeros(images.size(2), images.size(0), encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if gpu_mode else encoder_outputs

        for row in range(images.size(2)):
            image_input = images[:, :, row:row + 1, :].data.long()
            image_input = image_input.view(images.size(0), -1)
            encoder_output, encoder_hidden = encoder(image_input, encoder_hidden)
            encoder_outputs[row] = encoder_output[0]

        decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]] * images.size(0)))
        decoder_hidden = encoder_hidden
        decoder_input = decoder_input.cuda() if gpu_mode else decoder_input

        decoded_words = []
        #decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(topi)
            decoder_input = Variable(torch.LongTensor(topi))
            decoder_input = decoder_input.cuda() if gpu_mode else decoder_input


        for pos in range(len(decoded_words)):
            for batch in range(len(decoded_words[pos])):
                confusion_meter[labels[batch][pos]][decoded_words[pos][batch][0]] += 1


    print('Hom\thet\thom-alt\tSOS\tEOS')
    for i in range(len(confusion_meter)):
        print('\t'.join([str(x) for x in confusion_meter[i].tolist()]))



def train(train_file, batch_size, epoch_limit, file_name, gpu_mode):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    train_dataset = PileupDataset(train_file, transformations)
    train_loader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=gpu_mode # CUDA only
                             )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    encoder = EncoderRNN(batch_size, coverage*3, hidden_size=hidden_size, n_layers=n_layers)
    decoder = AttnDecoderRNN(batch_size, hidden_size, classN, n_layers=3, dropout_p=0.1, max_length=51)

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.001)

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(epoch_limit):
        total_loss = 0
        total_images = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            images = images.cuda() if gpu_mode else images
            labels = labels.cuda() if gpu_mode else labels

            encoder_hidden = encoder.initHidden(images.size(0))
            encoder_outputs = Variable(torch.zeros(images.size(2), images.size(0), encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if gpu_mode else encoder_outputs

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = 0

            for row in range(images.size(2)):
                image_input = images[:, :, row:row+1, :].data.long()
                image_input = image_input.view(images.size(0), -1)
                encoder_output, encoder_hidden = encoder(image_input, encoder_hidden)
                encoder_outputs[row] = encoder_output[0]

            decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]*images.size(0)))
            decoder_hidden = encoder_hidden
            decoder_input = decoder_input.cuda() if gpu_mode else decoder_input

            use_teacher_forcing = True

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(labels.size(1)):
                    target = labels[:, di]

                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target)
                    decoder_input = target  # Teacher forcing
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.data[0]
            total_images += images.size(2)
            print('Loss: ', loss.data[0]/images.size(2))
        print('Epoch average loss: ', total_loss/total_images)

    evaluate(train_file, batch_size, gpu_mode, encoder, decoder)


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
        default='./CRNN',
        help="Path and filename to save model, default is ./CNN"
    )
    parser.add_argument(
        "--gpu_mode",
        type=str,
        required=False,
        default=False,
        help="Cuda is on if true"
    )
    FLAGS, unparsed = parser.parse_known_args()

    train(FLAGS.csv_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out, FLAGS.gpu_mode)


