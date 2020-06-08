# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import argparse

import numpy as np

import torch
import MinkowskiEngine as ME

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from examples.resnet import *
import torch.optim as optim
from torch.autograd import Variable

class MnistDataLoader(DataLoader):

    class ToSparse(object):

        def __init__(self):
            pass

        def __call__(self, img):
            """
            Args:
                img (PIL Image): Image to be scaled.

            Returns:

            """
            arr = np.asarray(img)
            #
            # filled = arr.nonzero()
            # feat = arr[filled]
            #
            # return np.vstack((filled[0], filled[1], feat)).T

            # An intuitive way to extract coordinates and features
            coords, feats = [], []
            for i, row in enumerate(arr):
                for j, val in enumerate(row):
                    if val != 0:
                        coords.append([i, j])
                        feats.append([val])

            return torch.IntTensor(coords), torch.FloatTensor(feats)


        def __repr__(self):
            return self.__class__.__name__


    # Warning: read using mutable obects for default input arguments in python.
    def __init__(self, data_root, train=True, **kwargs):

        def collation(data):
            l = [(coords, feats, label) for ((coords, feats), label) in data]

            return ME.utils.batch_sparse_collate(l),

        super(MnistDataLoader, self).__init__(
            datasets.MNIST(root=data_root, download=True, train=train,
                           transform=transforms.Compose([
                               self.ToSparse(),
                           ])
                           ),
            shuffle=True,

            collate_fn=collation,
            **kwargs
        )



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

torch.cuda.manual_seed(0)


def eval(net, criterion):
    # print(
    #     f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
    # )

    val_loader = MnistDataLoader(data_root='/tmp/public_dataset/pytorch/', batch_size=1000, train=False,
                                 num_workers=8)

    net.eval()
    test_loss = 0
    correct = 0
    val_iter = 0
    print('Evaluating: ')
    for _, ll in enumerate(val_loader):
        (coords, feats, labels) = ll[0]
        labels = torch.IntTensor(labels)

        # indx_target = target.clone()

        out = net(ME.SparseTensor(feats.float(), coords).to(device))

        labels = labels.to(device)

        loss = criterion(out.F.squeeze(), labels.long())
        test_loss += loss.item()
        pred = out.F.max(1)[1]  # get the index of the max log-probability

        correct += pred.eq(labels).sum()
        val_iter = val_iter + 1

    test_loss = test_loss / len(val_loader)  # average over number of mini-batch
    acc = 100. * correct / len(val_loader.dataset)
    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(val_loader.dataset), acc))

    net.train()

    # if acc > best_acc:
    #     new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
    #     misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
    #     best_acc = acc
    #     old_file = new_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    config = parser.parse_args()


    train_loader = MnistDataLoader(data_root='/tmp/public_dataset/pytorch/', batch_size=1000, train=True, num_workers=8)


    net = ResNet14(
        1,  # in nchannel
        10,  # out_nchannel
        D=2)

    net = net.to(device)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)


    accum_loss, accum_iter, tot_iter = 0, 0, 0

    for epoch in range(config.max_epochs):

        net.train()

        # Training
        for batch_idx, l in enumerate(train_loader):

            (coords, feats, labels) = l[0]
            labels = torch.IntTensor(labels)

            # coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            # coords, feats, labels = Variable(coords), Variable(feats), Variable(labels)
            input = ME.SparseTensor(feats, coords).to(device)
            out = net(input)

            optimizer.zero_grad()

            loss = criterion(out.F.squeeze(), labels.to(device).long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 10 == 0 or tot_iter == 1:
                print(
                    f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
                )

            # if tot_iter % 500 == 0:
            #     eval(net, criterion)

        eval(net, criterion)







        pass

