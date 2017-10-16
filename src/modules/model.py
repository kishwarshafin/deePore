import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, inChannel=1, outChannel=256, coverageDepth=34, hidden_number=200, hidden_layer=3, classN=3, bidirection=False):
        super(CNN, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.coverageDepth = coverageDepth
        self.classN = classN
        self.hidden_number = hidden_number
        self.hidden_layer = hidden_layer
        self.direction = 1 if not bidirection else 2
        # -----CNN----- #
        self.batchNorm = nn.BatchNorm2d(outChannel)
        self.incpConv0 = nn.Conv2d(inChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.incpConv1 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.conv0 = nn.Conv2d(inChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.conv1 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.conv2 = nn.Conv2d(outChannel, outChannel, (1, 3), padding=(0, self.coverageDepth * 3), bias=False,
                               stride=(1, 3))
        self.conv3 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False)
        # -----RNN----- #
        self.lstm1 = nn.LSTM(coverageDepth * 3, hidden_number, hidden_layer, bidirectional=bidirection)
        # -----FCL----- #
        self.fc1 = nn.Linear(self.hidden_number * self.direction, self.hidden_number)
        self.fc2 = nn.Linear(self.hidden_number, self.classN * 10)
        self.fc3 = nn.Linear(self.classN * 10, self.classN)
        self.fc4 = nn.LogSoftmax()

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
        return (Variable(weight.new(self.direction * self.hidden_layer, seq_len, self.hidden_number).zero_()),
                Variable(weight.new(self.direction * self.hidden_layer, seq_len, self.hidden_number).zero_()))

    def lstm_layer(self, input):
        hidden = self.init_hidden(input.size(2))
        for i in range(input.size(0)):
            out, hidden = self.lstm1(input[i], hidden)
            hidden = self.repackage_hidden(hidden)
        return out

    def fully_connected_layer(self, x):
        batch_size = x.size(1)
        seq_length = x.size(0)
        x = x.view([batch_size * seq_length, -1])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view([batch_size, seq_length, self.classN])
        return x

    def forward(self, x):
        x = self.residual_layer(x, layer=0, batch_norm_flag=True)
        x = self.residual_layer(x, layer=1)
        x = self.residual_layer(x, layer=2)
        x = self.residual_layer(x, layer=3)
        x = self.residual_layer(x, layer=4)

        x = x.permute(1, 2, 0, 3)

        x = self.lstm_layer(x)
        x = self.fully_connected_layer(x)
        x = self.fc4(x)
        return x.view(-1, 3)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
