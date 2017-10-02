import argparse
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from scipy import misc
from modules.model import CNN
from modules.dataset import PileupDataset, TextColor
import sys


def test(csvFile, batchSize, modelPath):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    test_dset = PileupDataset(csvFile, transformations)
    testloader = DataLoader(test_dset,
                            batch_size=batchSize,
                            shuffle=False,
                            num_workers=4
                            # pin_memory=True # CUDA only
                            )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    cnn = torch.load(modelPath)
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    correct = 0
    total = 0
    total_hom = 0
    total_het = 0
    total_homalt = 0
    correct_hom = 0
    correct_het = 0
    correct_homalt = 0

    for counter, (images, labels) in enumerate(testloader):
        images = Variable(images)
        pl = labels

        for row in range(images.size(2)):
            x = images[:, :, row:row + 1, :]
            ypl = pl[:, row]
            outputs = cnn(x)

            _, predicted = torch.max(outputs.data, 1)
            for i, target in enumerate(ypl):
                t_tensor = torch.LongTensor([ypl[i]])
                p_tensor = torch.LongTensor([predicted[i]])
                if target == 0:
                    total_hom += 1
                    eq = torch.equal(t_tensor, p_tensor)
                    if eq:
                        correct_hom += 1
                        correct += 1
                elif target == 1:
                    total_het += 1
                    eq = torch.equal(t_tensor, p_tensor)
                    if eq:
                        correct_het += 1
                        correct += 1
                elif target == 2:
                    total_homalt += 1
                    eq = torch.equal(t_tensor, p_tensor)
                    if eq:
                        correct_homalt += 1
                        correct += 1
                total += 1

        if (counter+1) % 100 == 0:
            sys.stderr.write(TextColor.BLUE + " Batches done: " + str(counter+1) + TextColor.END)

    print('Total hom: ', total_hom, 'Correctly predicted: ', correct_hom, 'Accuracy: ', correct_hom / total_hom * 100)
    print('Total het: ', total_het, 'Correctly predicted: ', correct_het, 'Accuracy: ', correct_het / total_het * 100)
    print('Total homalt: ', total_homalt, 'Correctly predicted: ', correct_homalt, 'Accuracy: ',
          correct_homalt / total_homalt * 100)

    print("Test Accuracy of the model on the test images:", (100 * correct / total))
    print("Most populated class: ", total_hom/total * 100)



if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Testing data description csv file.."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size for testing, default is 100."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='./CNN.pkl',
        help="Saved model path."
    )
    parser.add_argument(
        "--gpu_mode",
        type=bool,
        default=False,
        help="If true then cuda is on."
    )
    FLAGS, unparsed = parser.parse_known_args()

    test(FLAGS.csv_file, FLAGS.batch_size, FLAGS.model_path)


