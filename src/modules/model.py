import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


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
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        self.rnn.flatten_parameters()
        return x


class CNN(nn.Module):
    def __init__(self, input_channel=1, output_channel=256, coverage_depth=34, hidden_size=200, hidden_layer=3, class_n=3, bidirectional=True):
        super(CNN, self).__init__()
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
        self.conv2 = nn.Conv2d(output_channel, output_channel, (1, 3), padding=(0, self.coverage_depth * 3), bias=False, stride=(1, 3))
        self.conv3 = nn.Conv2d(output_channel, output_channel, (1, 1), bias=False)
        # -----RNN----- #
        rnn_input_size = coverage_depth * 3 * output_channel
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
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, class_n, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_log_softmax = nn.LogSoftmax()

    def residual_layer(self, input_data, layer, batch_norm_flag=False):
        incpConv = self.incpConv1 if layer != 0 else self.incpConv0
        conv1 = self.conv1 if layer != 0 else self.conv0

        indataCp = self.batchNorm(incpConv(input_data)) if batch_norm_flag else incpConv(input_data)

        convOut1 = self.batchNorm(F.relu(conv1(input_data)))
        convOut2 = self.batchNorm(F.relu(self.conv2(convOut1)))
        convOut3 = self.conv3(convOut2)
        x = F.relu(indataCp + convOut3)
        return x

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

    def fully_connected_layer(self, x):
        batch_size = x.size(1)
        seq_length = x.size(0)
        x = x.view([batch_size * seq_length, -1])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view([batch_size, seq_length, self.class_n])
        return x

    def forward(self, x):
        #print('Input:', x.size())
        x = self.residual_layer(x, layer=0, batch_norm_flag=True)
        x = self.residual_layer(x, layer=1)
        x = self.residual_layer(x, layer=2)
        x = self.residual_layer(x, layer=3)
        x = self.residual_layer(x, layer=4)
        #print('After CNN: ', x.size())
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[3], sizes[2])
        sizes = x.size()
        #print(x.size())
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        #print('Reshape: ', x.size())
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        #print('Before RNN:', x.size())
        #exit()

        #x = x.permute(1, 2, 0, 3)
        x = self.rnns(x)
        #print('After RNN:', x.size())

        x = self.fc(x)
        #print('After FC:', x.size())
        #x = self.inference_log_softmax(x)
        #print('Final', x.size())
        #exit()
        return x.view(-1, 3)
