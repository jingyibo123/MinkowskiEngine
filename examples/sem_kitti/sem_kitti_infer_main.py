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
import os
from os import path

import sys


from examples.sem_kitti.trainer import Trainer

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import numpy as np
import torch
import MinkowskiEngine as ME

from examples.sem_kitti.tictoc import TicToc
from examples.minkunet import MinkUNet34C


if __name__ == '__main__':

    scan_file = '/media/yibo/DATA/data/KITTI/odometry/dataset/sequences/08/velodyne/000099.bin'
    model = '/mnt/fusion/yibo/source/MinkowskiEngine/results/MinkUNet32C/check_point_epoch_0_batch_4000'
    # scan_file = '/home/dev/yibo/data/KITTI/odometry/dataset/sequences/08/velodyne/000099.bin'
    # model = '/home/dev/yibo/source/MinkowskiEngine/results/MinkUNet32C/check_point_epoch_0_batch_4000'

    torch.cuda.manual_seed(0)

    device = 0

    with torch.no_grad():

        t = TicToc()

        # open and obtain scan
        scan = np.fromfile(scan_file, dtype=np.float32)

        print(t.toc() + 'read file')

        scan = scan.reshape(-1, 4)

        print(t.toc() + 'reshape')

        # label = np.fromfile(label_file, dtype=np.uint32)
        #
        # label = label & 0xFFFF  # semantic label in lower half
        # label = remap_lut[label]
        coords = scan[:, 0:3]
        int_coords = np.floor(coords / 0.01)
        feats = scan[:, 3:]

        print(t.toc() + 'multiply and slice')

        ind, ind_inv = ME.utils.sparse_quantize(int_coords,
                                                ignore_label=-100,
                                                return_index=True,
                                                return_inverse=True,
                                                quantization_size=0.01)

        print(t.toc() + ' sparse quantize, pts: ' + str(len(ind)))

        quantized_coords = int_coords[ind]
        quantized_feats = feats[ind]
        # quantized_labels = label[ind]

        # return ME.SparseTensor(
        #     feats=torch.from_numpy(scan[:, 3:]),
        #     coords=ME.utils.batched_coordinates([scan[:, 0:3] / voxel_size]),
        #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        # )

        quantized_coords = torch.IntTensor(quantized_coords)
        quantized_feats = torch.FloatTensor(quantized_feats)

        print(t.toc() + ' convert to torch tensor')

        quantized_coords, quantized_feats = ME.utils.sparse_collate([quantized_coords], [quantized_feats])

        print(t.toc() + ' sparse collate')

        tensor = ME.SparseTensor(quantized_feats, quantized_coords)

        print(t.toc() + ' to sparse tensor')

        net = MinkUNet34C(1, 20, D=3)

        print(t.toc() + ' construct net')

        net.load_state_dict(torch.load(model))

        print(t.toc() + ' load net')

        tensor = tensor.to(device)
        # torch.cuda.synchronize(device)
        print(t.toc() + ' upload data to cuda')

        net = net.to(device)
        # torch.cuda.synchronize(device)
        print(t.toc() + ' upload net to cuda')

        out = net(tensor)
        # torch.cuda.synchronize(device)
        print(t.toc() + ' infer')

        out = out.cpu()
        print(t.toc() + ' download result')


