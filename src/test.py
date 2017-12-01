import argparse
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from scipy import misc
from modules.dataset import PileupDataset, TextColor
import sys
import torchnet.meter as meter


def test(test_file, batch_size, model_path, gpu_mode, seq_len, num_classes=4):
    transformations = transforms.Compose([transforms.ToTensor()])

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    test_dset = PileupDataset(test_file, transformations)
    testloader = DataLoader(test_dset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=gpu_mode # CUDA only
                            )

    sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    model = torch.load(model_path)
    if gpu_mode:
        model = model.cuda()
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    confusion_matrix = meter.ConfusionMeter(num_classes)
    for counter, (images, labels, image_name) in enumerate(testloader):
        images = Variable(images, volatile=True)
        print(image_name)
        exit()
        pl = labels
        if gpu_mode:
            images = images.cuda()

        for row in range(images.size(2)):
            x = images[:, :, row:row+1, :]
            y = labels[:, row]
            preds = model(x)
            confusion_matrix.add(preds.data.squeeze(), y.type(torch.LongTensor))
            # print(y)
            # print(preds.data.squeeze())
        print(confusion_matrix.conf)
    print(confusion_matrix.conf)



if __name__ == '__main__':
    '''
    Processes arguments and performs tasks to generate the pileup.
    '''
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--test_file",
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
    parser.add_argument(
        "--seq_len",
        type=int,
        required=False,
        default=10,
        help="Sequences to see for each prediction."
    )
    FLAGS, unparsed = parser.parse_known_args()

    test(FLAGS.test_file, FLAGS.batch_size, FLAGS.model_path, FLAGS.gpu_mode, FLAGS.seq_len)


