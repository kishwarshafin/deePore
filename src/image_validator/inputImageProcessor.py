from pysam import VariantFile
import argparse
import os
import argparse
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from timeit import default_timer as timer
import sys
import torchvision
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PileupDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def summaryFileReader(self, summaryFile):
        f = open(summaryFile, "r")
        vcf_rec = ""
        imList = []
        i = 1
        class_no = 0
        cnt = {}
        classIds = {}
        for line in f:
            line = line.rstrip()
            if i % 2 == 0:
                recSplit = vcf_rec.split('\t')
                label = recSplit[-1].split(':')[0].rstrip()
                if str(label)!="0|1" and str(label)!="1|0" and str(label)!="1|1":
                    i+=1
                    continue
                if label not in cnt.keys():
                    classIds[label] = class_no + 1
                    cnt[label] = 1
                    class_no += 1
                else:
                    cnt[label] += 1
                file_path = line
                fpList = file_path.split(' ')
                imageFile = fpList[2]
                imList.append((imageFile, int(classIds[label])))
            else:
                vcf_rec = line
            i+=1
        for key in cnt.keys():
            print(key, cnt[key], classIds[key])
        return imList

    def __init__(self, summaryFileName, transform=None, target_transform=None):
        self.imList = self.summaryFileReader(summaryFileName)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        impath, target = self.imList[index]
        img = Image.open(os.path.join("../", impath + ".bmp")).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imList)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (6, 3))
        self.pool = nn.MaxPool2d((6, 3))
        self.conv2 = nn.Conv2d(6, 16, (6, 3))
        self.fc1 = nn.Linear(16 * 1 * 4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

    def forward(self, x):
        #print("BEFORE: ", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        #print("AFTER : ", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print("AFTER : ", x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print("AFTER : ", x.size())
        x = F.relu(self.fc1(x))
        #print("AFTER : ", x.size())
        x = F.relu(self.fc2(x))
        #print("AFTER : ", x.size())
        x = self.fc3(x)
        #print("AFTER : ", x.size())

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--summary_file_train",
        type=str,
        required=True,
        help="Summary file for train data."
    )
    parser.add_argument(
        "--summary_file_test",
        type=str,
        required = True,
        help="Summary file for test data."
    )
    FLAGS, unparsed = parser.parse_known_args()

    transformations = transforms.Compose([transforms.ToTensor()])
    train_dset = PileupDataset(FLAGS.summary_file_train, transformations)
    test_dset = PileupDataset(FLAGS.summary_file_test, transformations)
    trainloader = DataLoader(train_dset,
                              batch_size=2000,
                              shuffle=True,
                              num_workers=4
                              # pin_memory=True # CUDA only
                          )
    trainloader = DataLoader(test_dset,
                             batch_size=2000,
                             shuffle=False,
                             num_workers=4
                             # pin_memory=True # CUDA only
                             )
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #for batch_idx, (data, target) in enumerate(trainloader):
        #print(batch_idx)
    #for i in range(len(dset)):
        #img, label = dset[i]
        #npimg = img.numpy()
        #plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        #plt.show()
        #print(i, img.size(), label)
        #break

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))

    print('Finished Training')

    correct = 0
    total = 0
    for data in trainloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 230 test images: %d %%' % (
        100 * correct / total))

