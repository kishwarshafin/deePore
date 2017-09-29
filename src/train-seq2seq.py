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

hidden_size = 200
coverage = 34
classN = 5
window_size = 51
SOS_TOKEN = 4
EOS_token = 5
def train(csvFile, batchSize, epochLimit, fileName):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    trainDset = PileupDataset(csvFile, transformations)
    trainLoader = DataLoader(trainDset,
                             batch_size=batchSize,
                             shuffle=True,
                             num_workers=4
                             # pin_memory=True # CUDA only
                             )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    encoder = EncoderRNN(batchSize, coverage*3, hidden_size)
    decoder = AttnDecoderRNN(batchSize, hidden_size, classN, n_layers=3, dropout_p=0.1, max_length=51)

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.001)

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    for epoch in range(epochLimit):
        total_loss = 0
        total_images = 0
        for i, (images, labels) in enumerate(trainLoader):
            images = Variable(images)
            labels = Variable(labels)
            encoder_hidden = encoder.initHidden(images.size(0))
            encoder_outputs = Variable(torch.zeros(images.size(2), images.size(0), encoder.hidden_size))
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

            use_teacher_forcing = False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(labels.size(1)):
                    target = labels[:, di]
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target)
                    decoder_input = target  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(labels.size(1)):
                    target = labels[:, di]
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    decoder_input = decoder_input

                    loss += criterion(decoder_output, target)
                    if ni == EOS_token:
                        break
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            print('Loss: ', loss.data[0]/images.size(2))


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
    FLAGS, unparsed = parser.parse_known_args()

    train(FLAGS.csv_file, FLAGS.batch_size, FLAGS.epoch_size, FLAGS.model_out)


