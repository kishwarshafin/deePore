import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, inChannel, coverageDepth, classN, window_size, leak_value):
        super(CNN, self).__init__()
        self.inChannel = inChannel
        self.coverageDepth = coverageDepth
        self.classN = classN
        self.leak_value = leak_value
        self.outChannels = [self.inChannel, 32, 128, 256, 512, 1024]
        # -----CNN----- #
        self.identity1 = nn.Sequential(
            nn.Conv2d(self.outChannels[0], self.outChannels[1], (1, 1), bias=False, stride=(1, 1)),
        )
        self.cell1 = nn.Sequential(
            nn.Conv2d(self.outChannels[0], self.outChannels[1], (1, 1), bias=False, stride=(1, 1)),
            nn.BatchNorm2d(self.outChannels[1]),
            nn.LeakyReLU(self.leak_value),
            nn.Conv2d(self.outChannels[1], self.outChannels[1], (1, 5), padding=(0, 2), bias=False, stride=(1, 1)),
        )

        self.identity2 = nn.Sequential(
            nn.Conv2d(self.outChannels[1], self.outChannels[2], (1, 1), bias=False, stride=(1, 1))
        )
        self.cell2 = nn.Sequential(
            nn.Conv2d(self.outChannels[1], self.outChannels[2], (1, 1), bias=False, stride=(1, 1)),
            nn.BatchNorm2d(self.outChannels[2]),
            nn.LeakyReLU(self.leak_value),
            nn.Conv2d(self.outChannels[2], self.outChannels[2], (1, 5), padding=(0, 2), bias=False, stride=(1, 1)),
        )

        self.identity3 = nn.Sequential(
            nn.Conv2d(self.outChannels[2], self.outChannels[3], (1, 1), bias=False, stride=(1, 1))
        )
        self.cell3 = nn.Sequential(
            nn.Conv2d(self.outChannels[2], self.outChannels[3], (1, 1), bias=False, stride=(1, 1)),
            nn.BatchNorm2d(self.outChannels[3]),
            nn.LeakyReLU(self.leak_value),
            nn.Conv2d(self.outChannels[3], self.outChannels[3], (1, 5), padding=(0, 2), bias=False, stride=(1, 1)),
        )

        # self.identity4 = nn.Sequential(
        #     nn.Conv2d(self.outChannels[3], self.outChannels[4], (1, 1), bias=False, stride=(1, 1))
        # )
        # self.cell4 = nn.Sequential(
        #     nn.Conv2d(self.outChannels[3], self.outChannels[4], (1, 1), bias=False, stride=(1, 1)),
        #     nn.BatchNorm2d(self.outChannels[4]),
        #     nn.LeakyReLU(self.leak_value),
        #     nn.Conv2d(self.outChannels[4], self.outChannels[4], (1, 5), padding=(0, 2), bias=False, stride=(1, 1)),
        # )
        #
        # self.identity5 = nn.Sequential(
        #     nn.Conv2d(self.outChannels[4], self.outChannels[5], (1, 1), bias=False, stride=(1, 1))
        # )
        # self.cell5 = nn.Sequential(
        #     nn.Conv2d(self.outChannels[4], self.outChannels[5], (1, 1), bias=False, stride=(1, 1)),
        #     nn.BatchNorm2d(self.outChannels[5]),
        #     nn.LeakyReLU(self.leak_value),
        #     nn.Conv2d(self.outChannels[5], self.outChannels[5], (1, 5), padding=(0, 2), bias=False, stride=(1, 1)),
        # )

        # -----FCL----- #
        self.fc1 = nn.Linear(self.outChannels[3] * coverageDepth, self.classN)
        # self.fc2 = nn.Linear(1000, self.classN)
        self.fc3 = nn.LogSoftmax()

    def residual_layer(self, input_data, identity, cell):
        ix = identity(input_data)
        x = cell(input_data)
        LR = nn.LeakyReLU(self.leak_value)
        return LR(x + ix)

    def fully_connected_layer(self, x):
        batch_size = x.size(0)
        x = x.view([batch_size, -1])
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.residual_layer(x, self.identity1, self.cell1)
        x = self.residual_layer(x, self.identity2, self.cell2)
        x = self.residual_layer(x, self.identity3, self.cell3)
        # x = self.residual_layer(x, self.identity4, self.cell4)
        # x = self.residual_layer(x, self.identity5, self.cell5)

        x = self.fully_connected_layer(x)

        return x.view(-1, self.classN)
