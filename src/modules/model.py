import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

use_cuda = False

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.GRU, bidirectional=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, input, h_input):
        print('-----_HERE_--------')
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(input, h_input)
        if self.bidirectional:
            # (TxNxH*2) -> (TxNxH) by sum
            output = output.view(output.size(0), output.size(1), 2, -1).sum(2).view(output.size(0), output.size(1), -1)

        return output, hidden

class EncoderCRNN(nn.Module):
    def __init__(self, input_channel=1, output_channel=256, coverage_depth=34, hidden_size=500, hidden_layer=5, class_n=3, bidirectional=True):
        super(EncoderCRNN, self).__init__()
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
        self.embedding = nn.Embedding(output_channel * 3 * coverage_depth, hidden_size)
        self.gruInitial = nn.GRU(output_channel * 3 * coverage_depth, hidden_size)
        self.gruRecurrent = nn.GRU(hidden_size, hidden_size)

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

    def forward(self, x, hidden):
        x = self.residual_layer(x, layer=0, batch_norm_flag=True)
        x = self.residual_layer(x, layer=1)
        x = self.residual_layer(x, layer=2)
        x = self.residual_layer(x, layer=3)
        x = self.residual_layer(x, layer=4)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[3], sizes[2])
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        #x = x.view(x.size(0), -1)

        #print(x.size())
        #print(type(x))
        #embedded = self.embedding(Variable(x.data.long())).view(1, x.size(0), -1)
        #output = embedded
        #print(output.size())
        #exit()
        output, hidden = self.gruInitial(x, hidden)
        for i in range(self.hidden_layer):
            output, hidden = self.gruRecurrent(output, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.hidden_layer * self.direction, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class EncoderRNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, n_layers=3):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.directions = 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gruInitial = nn.GRU(input_size * hidden_size, hidden_size)
        self.gruRecurrent = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        self.batch_size = input.size(0)
        embedded = self.embedding(Variable(input)).view(1, self.batch_size, -1)
        output = embedded
        output, hidden = self.gruInitial(output, hidden)
        for i in range(self.n_layers):
            output, hidden = self.gruRecurrent(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers * self.directions, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


####-----DECODER START-----#####
class AttnDecoderRNN(nn.Module):
    def __init__(self, batch_size, hidden_size, output_size, n_layers=3, dropout_p=0.1, max_length=51):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #print('Decoder')
        self.batch_size = input.size(0)
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_weights = attn_weights.view([self.batch_size, 1, -1])
        encoder_outputs = encoder_outputs.view([self.batch_size, self.max_length, -1])
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        embedded = embedded.view([self.batch_size, 1, -1])
        output = torch.cat((embedded, attn_applied), 2).view([self.batch_size, -1])
        output = self.attn_combine(output).unsqueeze(0)


        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x, h):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x, h

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


