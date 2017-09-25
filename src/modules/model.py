import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
#class RNN(nn.Module):

class CRNN(nn.Module):
    def __init__(self, inChannel=1, outChannel=256, coverageDepth=102, hiddenNum=200, hiddenLayer=3, bidirection=True, classN=3):
        super(CRNN, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.coverageDepth = coverageDepth
        self.hiddenNum = hiddenNum
        self.hiddenLayer = hiddenLayer
        self.direction = 2
        self.classN = classN
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
        #####-----FCL-----#####
        self.fc1 = nn.Linear(self.hiddenNum * self.direction, self.hiddenNum)
        self.fc2 = nn.Linear(self.hiddenNum, self.classN * 10)
        self.fc3 = nn.Linear(self.classN * 10, self.classN)

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

    def lstmLayer(self, input):
        hx = Variable(torch.randn(self.direction * self.hiddenLayer, input.size(2), self.hiddenNum))
        cx = Variable(torch.randn(self.direction * self.hiddenLayer, input.size(2), self.hiddenNum))
        out = Variable(torch.randn(input.size(1), input.size(2), self.hiddenNum))
        hidden = (hx, cx)
        for i in range(input.size(0)):
            out, hidden = self.lstm1(input[i], hidden)

        return out

    def fullyConnectedLayer2(self, lasth):
        batch_size = lasth.size(1)
        max_time = lasth.size(0)
        x = lasth.view([batch_size*max_time, -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view([batch_size, max_time, self.classN])
        #print(x.size())
        return x

    def fullyConnectedLayer(self, lasth):
        batch_size = lasth.size(1)
        max_time = lasth.size(0)
        weight_out = Variable(
            self.get_truncated_normal(self.direction * self.hiddenNum, np.sqrt(2.0 / (self.direction * self.hiddenNum))).view(
                [self.direction, self.hiddenNum]))
        weight_class = Variable(
            self.get_truncated_normal(self.hiddenNum * self.classN, np.sqrt(2.0 / (self.hiddenNum))).view([self.hiddenNum, self.classN]))
        lasth_rs = lasth.view([batch_size * max_time, self.hiddenNum, self.direction])
        lasth_output = torch.matmul(lasth_rs, weight_out).sum(2)
        logits = torch.matmul(lasth_output, weight_class)
        logits = logits.view([batch_size, max_time, self.classN])
        return logits

    def get_truncated_normal(self, size, stdv):
        return torch.Tensor(size).normal_(std=stdv).clamp_(min=-2*stdv, max=2*stdv)

    def forward(self, x):
        #print('Input: ',x.size())
        R1 = self.residualLayer(x, layer=0, batchNormFlag=True)
        R2 = self.residualLayer(R1, layer=1)
        R3 = self.residualLayer(R2, layer=2)
        #print('CNN', R3.size())
        RNN_in = R3.permute(1, 2, 0, 3)
        #print('RNN in', RNN_in.size())
        RNN_out = self.lstmLayer(RNN_in)
        #print('LSTM', RNN_out.size())
        logits = self.fullyConnectedLayer2(RNN_out)
        #print('Logits: ', logits.size())
        return logits


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
