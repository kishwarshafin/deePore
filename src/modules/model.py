import torch.nn as nn
import torch.nn.functional as F


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (1, 1))
        self.pool = nn.MaxPool2d((1, 3))
        self.conv2 = nn.Conv2d(6, 16, (1, 1))
        self.pool2 = nn.MaxPool2d((1, 3))
        self.fc1 = nn.Linear(176, 44)
        self.fc2 = nn.Linear(44, 11)
        self.fc3 = nn.Linear(11, 3)

    def forward(self, x):
        # print("BEFORE: ", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print("AFTER : ", x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        # print("AFTER : ", x.size())
        x = x.view(-1, self.num_flat_features(x))
        # print("AFTER : ", x.size())
        x = F.relu(self.fc1(x))
        # print("AFTER : ", x.size())
        x = F.relu(self.fc2(x))
        # print("AFTER : ", x.size())
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
