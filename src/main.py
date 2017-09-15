import argparse
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MultiLabelBinarizer
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import CNN
from scipy import misc

class PileupDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, transform=None):
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_file'].apply(lambda x: os.path.isfile(x)).all(), \
            "Some images referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.transform = transform

        self.X_train = tmp_df['image_file']
        labelLists = []
        for label in tmp_df['label']:
            labelList = [int(x) for x in str(label)]
            labelLists.append(np.array(labelList, dtype=np.long))
        self.y_train = np.array(labelLists)

    def __getitem__(self, index):
        # print('FILE:', self.X_train[index])
        # print('INDX:', index)
        img = Image.open(self.X_train[index])
        img = ImageOps.grayscale(img)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


def train(summary_file, fileName):
    transformations = transforms.Compose([transforms.ToTensor()])
    train_dset = PileupDataset(summary_file, transformations)
    trainloader = DataLoader(train_dset,
                             batch_size=20,
                             shuffle=True,
                             num_workers=4
                             # pin_memory=True # CUDA only
                             )

    cnn = CNN()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Train the Model
    for epoch in range(100):
        total_loss = 0
        total_images = 0
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images)
            labels = Variable(labels)

            for row in range(images.size(2)):
                # segmentation of image. Currently using 1xCoverage
                x = images[:, :, row:row + 1, :]
                y = labels[:, row]

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = cnn(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # loss count
                total_images += 20  # batch_size
                total_loss += loss

        print('EPOCH: ', epoch, end='')
        print(' Image segments ', total_images, 'Avg Loss: ', total_loss.data[0] / total_images)

    print('Finished Training')
    torch.save(cnn, './model/CNN.pkl')

    # Save and load only the model parameters(recommended).
    torch.save(cnn.state_dict(), './model/params.pkl')


def test(summary_file, model_path):
    transformations = transforms.Compose([transforms.ToTensor()])
    test_dset = PileupDataset(summary_file, transformations)
    testloader = DataLoader(test_dset,
                            batch_size=20,
                            shuffle=False,
                            num_workers=4
                            # pin_memory=True # CUDA only
                            )
    # Test the Model
    cnn = torch.load(model_path)
    #cnn = cnn.load_state_dict(model_path)
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    total_hom = 0
    total_het = 0
    total_homalt = 0
    correct_hom = 0
    correct_het = 0
    correct_homalt = 0
    for images, labels in testloader:
        images = Variable(images)
        pl = labels
        labels = Variable(labels)
        # print(labels.size())
        for row in range(images.size(2)):
            # print(i, row, images[:, :, row:row+1, :].size(), labels[0][row])
            x = images[:, :, row:row + 1, :]
            y = labels[:, row]
            ypl = pl[:, row]
            outputs = cnn(x)

            _, predicted = torch.max(outputs.data, 1)
            # print(predicted.size())
            # print(ypl.size())
            for i, target in enumerate(ypl):
                if target == 0:
                    total_hom += 1
                    eq = (predicted[i] == target).sum()
                    if eq:
                        correct_hom += 1
                elif target == 1:
                    total_het += 1
                    eq = (predicted[i] == target).sum()
                    if eq:
                        correct_het += 1
                elif target == 2:
                    total_homalt += 1
                    eq = (predicted[i] == target).sum()
                    if eq:
                        correct_homalt += 1

            total += ypl.size(0)
            correct += (predicted == ypl).sum()
    print('Total hom: ', total_hom, 'Correctly predicted: ', correct_hom, 'Accuracy: ', correct_hom / total_hom * 100)
    print('Total het: ', total_het, 'Correctly predicted: ', correct_het, 'Accuracy: ', correct_het / total_het * 100)
    print('Total homalt: ', total_homalt, 'Correctly predicted: ', correct_homalt, 'Accuracy: ',
          correct_homalt / total_homalt * 100)

    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))



if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--summary_file",
        type=str,
        required = True,
        help="Summary file name."
    )
    parser.add_argument(
        "--train",
        type=bool,
        required=False,
        default=False,
        help="Train mode."
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=False,
        default='./model/CNN.pt',
        help="Path and filename to save model."
    )
    parser.add_argument(
        "--predict",
        type=bool,
        required=False,
        default=False,
        help="Predict mode."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./model/CNN.pt',
        help="Saved model path."
    )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.train:
        train(FLAGS.summary_file, FLAGS.model_out)
    elif FLAGS.predict:
        test(FLAGS.summary_file, FLAGS.model_path)
    else:
        print("CHOOSE EITHER TRAIN OR PREDICT MODE.")


