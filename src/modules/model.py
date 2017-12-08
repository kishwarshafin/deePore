import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Single block defining a single cell in the network
class SingleBlock(nn.Module):
    """
    Structure of a single block/cell in a layer. Consider this as a single neuron of the network.
    """
    def __init__(self, in_channels, out_channels, stride, leak_value=0.1, drop_rate=0.0):
        """
        Initialize a single cell.
        """
        super(SingleBlock, self).__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu1 = nn.LeakyReLU(leak_value)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(leak_value)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        """
        Perform all calculations that happen in a single cell block.
        """
        identity_conv = self.identity_conv(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return self.relu2(identity_conv + out)


class LayerBlock(nn.Module):
    """
    Structure of layer of the neural network. A layer would contain N number of blocks.
    """
    def __init__(self, nb_layers, in_channels, out_channels, block, stride, leak_value, drop_rate=0.0):
        """
        Initialize a layer. Block is a cell represented as an object of SingleBlock class
        """
        super(LayerBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, nb_layers, stride, leak_value, drop_rate)

    def _make_layer(self, block, in_channels, out_channels, nb_layers, stride, leak_value, drop_rate):
        """
        """
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels, i == 0 and stride or 1,
                                leak_value, drop_rate))
            return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    """
    The model itself. It's a three block model with variable depth and widen factor.
    """

    def __init__(self, input_channels, depth, num_classes, widen_factor, drop_rate, column_width, seq_len, leak_value):
        super(Model, self).__init__()
        nChannels = [10, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = SingleBlock
        self.nChannels = nChannels[3]
        self.num_classes = num_classes
        # 1st conv before any network block
        # 1st block
        self.block1 = LayerBlock(n, nChannels[0], nChannels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = LayerBlock(n, nChannels[1], nChannels[2], block, 1, drop_rate)
        # 3rd block
        self.block3 = LayerBlock(n, nChannels[2], nChannels[3], block, 1, drop_rate)
        # global average pooling and classifier
        self.fc = nn.Linear(nChannels[3] * column_width * seq_len, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block1.forward(x)
        # print(out.size())
        out = self.block2.forward(out)
        # print(out.size())
        out = self.block3.forward(out)
        # print(out.size())
        # out = self.relu(self.bn1(out))
        out = out.view(out.size(0), -1)
        # print("HERE", out.size())
        out = self.fc(out)
        # print(out.size())
        # exit()
        return out

"""
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, inChannel=1, outChannel=256, coverageDepth=34, classN=3, window_size=1):
        super(CNN, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.coverageDepth = coverageDepth
        self.classN = classN
        # -----CNN----- #
        self.batchNorm = nn.BatchNorm2d(outChannel)
        self.incpConv0 = nn.Conv2d(inChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.incpConv1 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.conv0 = nn.Conv2d(inChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.conv1 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False, stride=(1, 1))
        self.conv2 = nn.Conv2d(outChannel, outChannel, (1, 3), padding=(0, self.coverageDepth * 3), bias=False,
                               stride=(1, 3))
        self.conv3 = nn.Conv2d(outChannel, outChannel, (1, 1), bias=False)
        # -----FCL----- #
        self.fc1 = nn.Linear(outChannel * coverageDepth * 3 * window_size, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, self.classN)
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

    def fully_connected_layer(self, x):
        batch_size = x.size(0)
        x = x.view([batch_size, -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.residual_layer(x, layer=0, batch_norm_flag=True)
        x = self.residual_layer(x, layer=1)
        x = self.residual_layer(x, layer=2)
        x = self.residual_layer(x, layer=3)
        x = self.residual_layer(x, layer=4)

        x = self.fully_connected_layer(x)
        x = self.fc4(x)
        return x.view(-1, 3)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features"""
