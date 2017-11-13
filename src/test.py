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


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if h == Variable:
        return Variable(h.data)
    else:
        return repackage_hidden(h)

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

    model = torch.load(modelPath)
    if gpu_mode:
        model = model.cuda()
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    confusion_matrix = meter.ConfusionMeter(3)
    seq_len = 3
    confusion_tensor = torch.zeros(3, 3)
    for counter, (images, labels) in enumerate(testloader):
        images = Variable(images, volatile=True)
        hidden = model.init_hidden(images.size(0))
        pl = labels
        if gpu_mode:
            images = images.cuda()

        for row in range(0, images.size(2), seq_len):
            if row + seq_len > images.size(2):
                seq_len = images.size(2) - row
            else:
                seq_len = 3

            x = images[:, :, row:row + seq_len, :]
            ypl = pl[:, row:row+seq_len]
            # ypl = ypl.contiguous().view(-1)
            preds = model(x, hidden)
            # preds = preds.contiguous().view(-1, 3)

            for i in range(preds.size(0)):
                #predictions = torch.max(preds[i], 1)[1]
                targets = ypl[i]
                predictions = preds[i].data.topk(1)

                for i in range(predictions[1].size(0)):
                    p = predictions[1][i][0]
                    t = targets[i]
                    confusion_tensor[t][p] += 1
                    #if t != p:
                        #print(preds[i], targets)
            confusion_matrix.add(preds.contiguous().view(-1,3).data.squeeze(), ypl.contiguous().view(-1))
           #  print(confusion_tensor)
    print(confusion_tensor)
    print(confusion_matrix.conf)
    print(confusion_matrix)



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


