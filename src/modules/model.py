import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np


##===This part of the code is copied from deepspeech.pytorch module===#
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            batch_size = input_.size()[0]
            return torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Model(nn.Module):
    def __init__(self, input_channel, output_channel, coverage_depth,
                 hidden_size, hidden_layer, class_n, bidirectional, batch_size):
        super(Model, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.coverage_depth = coverage_depth
        self.classN = class_n
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.bidirectional = bidirectional
        self.direction = 1 if not bidirectional else 2
        # -----CNN----- #
        self.batchNorm = nn.BatchNorm2d(output_channel)
        self.incpConv0 = nn.Conv2d(input_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        self.incpConv1 = nn.Conv2d(output_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        self.conv0 = nn.Conv2d(input_channel, output_channel, (1, 3), padding=(0, 1), bias=False)
        self.conv1 = nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, 1), bias=False)
        self.conv2 = nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, 1), bias=True)
        self.conv3 = nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, 1), bias=True)
        # ----FC before CNN---- #s
        self.cnn_fc_rnn = nn.Linear(coverage_depth * output_channel, hidden_size)
        # -----RNN----- #
        rnn_type = nn.GRU
        rnn_input_size = hidden_size
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(hidden_layer - 1):
            rnn = BatchRNN(input_size=hidden_size, hidden_size=hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        # -----FCL----- #
        self.fc1 = nn.Linear(hidden_size, self.classN)
        self.inference_log_softmax = nn.LogSoftmax()

    def residual_layer(self, input_data, layer, batch_norm_flag=False):
        incpConv = self.incpConv1 if layer != 0 else self.incpConv0
        conv1 = self.conv1 if layer != 0 else self.conv0

        indataCp = self.batchNorm(incpConv(input_data)) if batch_norm_flag else incpConv(input_data)

        convOut1 = self.batchNorm(conv1(input_data))

        convOut2 = self.batchNorm(self.conv2(convOut1))
        convOut3 = self.conv3(convOut2)

        x = indataCp + convOut3
        return x

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.direction * self.hidden_layer, batch_size, self.hidden_size).zero_())

    def fully_connected_layer(self, x):
        # batch_size = x.size(0)
        # x = x.view([batch_size, -1])
        x = self.fc1(x)
        return x

    def rnn_layer(self, x):
        # print(x.size(), hidden.size())
        # hidden = self.init_hidden(x.size(0))
        for i, rnn in self.rnns:
            # rnn.flatten_parameters()
            # print(i, rnn)
            # print(x.size(), self.hidden.size())
            x = rnn(x)
            # hidden = h
            # self.hidden = hidden
            # print(x.size())
            # print(x.size())
            # if self.bidirectional:
                # (TxNxH*2) -> (TxNxH) by sum
                # x = x.contiguous().view(x.size(0), x.size(1), 2, -1)
                # x = x.sum(2)
                # x = x.view(x.size(0), x.size(1), -1)
            # print(x.size())
        # print(x.size())
        return x

    def forward(self, x):
        x = self.residual_layer(x, layer=0, batch_norm_flag=True)
        x = self.residual_layer(x, layer=1)
        x = self.residual_layer(x, layer=2)
        x = self.residual_layer(x, layer=3)
        x = self.residual_layer(x, layer=4)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[3], sizes[2])  # Collapse feature dimension
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        x = self.cnn_fc_rnn(x)
        # print(x.size())
        x = self.rnns(x)
        x = self.fully_connected_layer(x)
        # x = x.transpose(0, 1)
        return x
