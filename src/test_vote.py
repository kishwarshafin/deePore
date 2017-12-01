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


def most_common(lst):
    return max(set(lst), key=lst.count)


def test(csvFile, batchSize, modelPath, gpu_mode, seq_len, num_classes):
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

    model = torch.load(modelPath)
    if gpu_mode:
        model = model.cuda()
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    seq_len = seq_len
    confusion_tensor = torch.zeros(num_classes, num_classes)
    smry = open("out_" + csvFile.split('/')[-1], 'w')
    for counter, (images, labels, image_name) in enumerate(testloader):
        images = Variable(images, volatile=True)
        pl = labels
        if gpu_mode:
            images = images.cuda()
        window = 1
        prediction_stack = []
        for row in range(0, images.size(2), 1):

            if gpu_mode and images.size(0) % 8 != 0:
                continue

            x = images[:, :, row:row + seq_len, :]
            # print(x.size())
            # print(x.size())
            ypl = pl[:, row]
            preds = model(x)
            preds = preds.data.topk(1)[1]
            prediction_stack.append(preds)

            if row+1 >= seq_len:
                for i in range(images.size(0)):
                    pr = []
                    k = seq_len - 1
                    for j in range(len(prediction_stack)):
                        pr.append(prediction_stack[j][i][k][0])
                        k -= 1
                    p = most_common(pr)
                    t = ypl[i]
                    if t != p:
                        smry.write(str(t) + ',' + str(p) + ',' + str(pr) + ',' + image_name[i] + ',' + str(row) + "\n")
                    confusion_tensor[t][p] += 1
                prediction_stack.pop(0)
                window = 1
            window += 1
        print(confusion_tensor)
        #break

    smry.close()
    print(confusion_tensor)


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
        default=5,
        help="Sequences to see for each prediction."
    )
    FLAGS, unparsed = parser.parse_known_args()

    test(FLAGS.test_file, FLAGS.batch_size, FLAGS.model_path, FLAGS.gpu_mode, FLAGS.seq_len, num_classes=4)


