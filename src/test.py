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
import torchnet.meter as meter

def test(csvFile, batchSize, modelPath, gpu_mode):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    test_dset = PileupDataset(csvFile, transformations)
    testloader = DataLoader(test_dset,
                            batch_size=batchSize,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=gpu_mode # CUDA only
                            )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    cnn = torch.load(modelPath)
    if gpu_mode:
        cnn = cnn.cuda()
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    confusion_matrix = meter.ConfusionMeter(3)
    for counter, (images, labels) in enumerate(testloader):
        images = Variable(images, volatile=True)
        pl = labels
        if gpu_mode:
            images = images.cuda()

        for row in range(images.size(2)):
            x = images[:, :, row:row + 1, :]
            ypl = pl[:, row]
            preds = cnn(x)

            confusion_matrix.add(preds.data.squeeze(), ypl.type(torch.LongTensor))
    print(confusion_matrix.conf)



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

    test(FLAGS.csv_file, FLAGS.batch_size, FLAGS.model_path, FLAGS.gpu_mode)


