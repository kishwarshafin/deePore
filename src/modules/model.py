import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np


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


class InferenceBatchLogSoftmax(nn.Module):
    def forward(self, input_):
        batch_size = input_.size()[0]
        return torch.stack([F.log_softmax(input_[i]) for i in range(batch_size)], 0)


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm2d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        #print(x.size())
        x, _ = self.rnn(x)
        #print(x.size())
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        self.rnn.flatten_parameters()
        return x


class Model(nn.Module):
    def __init__(self, input_channel=3, output_channel=256, coverage_depth=50, hidden_size=500, hidden_layer=3, class_n=3, bidirectional=True):
        super(Model, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.coverage_depth = coverage_depth
        self.class_n = class_n
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.direction = 1 if not bidirectional else 2
        # -----CNN----- #
        self.batchNorm = nn.BatchNorm2d(output_channel)
        self.incpConv0 = nn.Conv2d(input_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        self.incpConv1 = nn.Conv2d(output_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        self.conv0 = nn.Conv2d(input_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        self.conv1 = nn.Conv2d(output_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        self.conv2 = nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, self.coverage_depth), bias=False, stride=(1, 3))
        self.conv3 = nn.Conv2d(output_channel, output_channel, (1, 1), bias=False)

        self.residual_layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, (1, 3), padding=(0, self.coverage_depth), bias=False,
                      stride=(1, 3)),
            nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, self.coverage_depth), bias=False,
                      stride=(1, 3)),
            nn.BatchNorm2d(output_channel)
        )
        self.identity = nn.Conv2d(input_channel, output_channel, (1, 1), bias=False, stride=(1, 1))

        self.residual_layer2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, self.coverage_depth), bias=False,
                      stride=(1, 3)),
            nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, self.coverage_depth), bias=False,
                      stride=(1, 3)),
            nn.BatchNorm2d(output_channel)
        )
        self.identity2 = nn.Conv2d(output_channel, output_channel, (1, 1), bias=False, stride=(1, 1))
        # -----RNN----- #
        rnn_input_size = coverage_depth * output_channel
        rnn_type = nn.LSTM
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
        fully_connected = nn.Sequential(
            nn.Linear(hidden_size, class_n)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_log_softmax = nn.LogSoftmax()

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def init_hidden(self, seq_len):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.direction * self.hidden_layer, seq_len, self.hidden_size).zero_()),
                Variable(weight.new(self.direction * self.hidden_layer, seq_len, self.hidden_size).zero_()))

    def forward(self, x):

        x_r = self.residual_layer(x)
        x_i = self.identity(x)
        x = F.relu(x_r+x_i)
        x_r = self.residual_layer2(x)
        x_i = self.identity2(x)
        x = F.relu(x_r + x_i)
        x_r = self.residual_layer2(x)
        x_i = self.identity2(x)
        x = F.relu(x_r + x_i)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[3], sizes[2])
        sizes = x.size()

        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        x = self.fc(x)
        return x.view(-1, 3)
