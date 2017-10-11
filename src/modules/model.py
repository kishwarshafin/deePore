import torch.nn as nn
import torch.nn.functional as F


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, inChannel=1, outChannel=256, coverageDepth=34, classN=3):
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
        self.fc1 = nn.Linear(outChannel * coverageDepth * 3, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, self.classN)

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
        convOut3 = self.conv3(convOut2)
        x = F.relu(indataCp + convOut3)
        return x

    def fullyConnectedLayer(self, x):
        batch_size = x.size(0)
        x = x.view([batch_size, -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.residualLayer(x, layer=0, batchNormFlag=True)
        x = self.residualLayer(x, layer=1)
        x = self.residualLayer(x, layer=2)

        x = self.fullyConnectedLayer(x)
        return x.cpu().view(-1, 3)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
