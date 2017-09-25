import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
#class RNN(nn.Module):

class CNN(nn.Module):
    def __init__(self, inChannel = 1, outChannel = 256, coverageDepth = 102, hiddenNum = 200, hiddenLayer = 3, bidirection = True):
        super(CNN, self).__init__()
        #####-----CNN-----#####
        self.batchNorm = nn.BatchNorm2d(outChannel)
        self.incpConv0 = nn.Conv2d(inChannel, outChannel, (1, 1), bias=False, stride=(1, 1, 1, 1))
        self.incpConv1 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1, 1, 1))
        self.conv0 = nn.Conv2d(inChannel, outChannel, (1, 1), bias=False, stride=(1, 1, 1, 1))
        self.conv1 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1, 1, 1))
        self.conv2 = nn.Conv2d(outChannel, outChannel, (3, 1), padding=(1, 0), bias=False, stride=(1, 1, 1, 1))
        self.conv3 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1, 1, 1))
        #####-----RNN-----#####
        self.lstm1 = nn.LSTM(coverageDepth, hiddenNum, hiddenLayer, bidirectional=bidirection)

    def residualLayer(self, indata, layer, batchNormFlag=False):
        incpConv = self.incpConv1
        conv1 = self.conv1
        if layer == 0:
            incpConv = self.incpConv0
            conv1 = self.conv0
        if batchNormFlag is True:
            indataCp = self.batchNorm(incpConv(indata))
        else:
            indataCp = incpConv(indata)
        convOut1 = self.batchNorm(F.relu(conv1(indata)))
        convOut2 = self.batchNorm(F.relu(self.conv2(convOut1)))
        convOut3 = self.batchNorm(self.conv3(convOut2))
        x = F.relu(indataCp+convOut3)
        return x

    def lstmLayer(self, input, hidden_num=200, hidden_layer=3, direction=2):
        hx = Variable(torch.randn(direction * hidden_layer, input.size(2), hidden_num))
        cx = Variable(torch.randn(direction * hidden_layer, input.size(2), hidden_num))
        out = Variable(torch.randn(input.size(1), input.size(2), hidden_num))
        hidden = (hx, cx)
        for i in range(input.size(0)):
            out, hidden = self.lstm1(input[i], hidden)

        return out


    def fullyConnectedLayer(self, lasth, direction = 2, hidden_num = 200, class_n = 3):
        batch_size = lasth.size(1)
        max_time = lasth.size(0)
        weight_out = Variable(
            self.get_truncated_normal(direction * hidden_num, np.sqrt(2.0 / (direction * hidden_num))).view(
                [direction, hidden_num]))
        weight_class = Variable(
            self.get_truncated_normal(hidden_num * class_n, np.sqrt(2.0 / (hidden_num))).view([hidden_num, class_n]))
        lasth_rs = lasth.view([batch_size * max_time, hidden_num, direction])
        lasth_output = torch.matmul(lasth_rs, weight_out).sum(2)
        logits = torch.matmul(lasth_output, weight_class)
        logits = logits.view([batch_size, max_time, class_n])
        return logits

    def get_truncated_normal(self, size, stdv):
        return torch.Tensor(size).normal_(std=stdv).clamp_(min=-2*stdv, max=2*stdv)

    def forward(self, x):
        print('Input: ',x.size())
        R1 = self.residualLayer(x, layer=0, batchNormFlag=True)
        R2 = self.residualLayer(R1, layer=1)
        R3 = self.residualLayer(R2, layer=2)
        print('CNN', R3.size())
        RNN_in = R3.permute(1, 2, 0, 3)
        print('RNN in', RNN_in.size())
        RNN_out = self.lstmLayer(RNN_in)
        print('LSTM', RNN_out.size())
        logits = self.fullyConnectedLayer(RNN_out)
        print('Logits: ', logits.size())
        return logits


    def num_flat_features(self, x):
        size = x.size()[2:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
