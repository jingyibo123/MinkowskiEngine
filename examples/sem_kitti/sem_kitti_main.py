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


from examples.sem_kitti.trainer import *

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences_dir', required=True, type=str)
    # parser.add_argument('--max_epochs', default=100, type=int)
    # parser.add_argument('--lr', default=0.1, type=float)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay', type=float, default=1e-4)

    import sys
    sys.argv.append('--sequences_dir')
    sys.argv.append('dd')

    config = parser.parse_args()
    '/home/dev/yibo/data/KITTI/odometry/dataset/sequences/'
    '/mnt/fusion/yibo/data/KITTI/odometry/dataset/sequences'


    trainer = Trainer(sequences_dir='/home/dev/yibo/data/KITTI/odometry/dataset/sequences/',
                      config_yaml='examples/sem_kitti/config/MinkUNet34C.yaml',
                      data_yaml='examples/sem_kitti/config/semantic-kitti.yaml',
                      save_dir='results/MinkUNet14A_old')

    trainer.train()


