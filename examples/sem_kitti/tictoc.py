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
import time


class TicToc:
    def __init__(self):
        self.start = 0.0
        self.tic()

    def tic(self):
        self.start = time.time()

    def toc(self):

        dur = time.time() - self.start
        if dur > 120:
            msg = f' Time elapsed: {dur/60.0:.2f} min: '
        if dur > 2:
            msg = f' Time elapsed: {dur:.2f} s: '
        elif dur > 2e-3:
            msg = f' Time elapsed: {dur*1e3:.2f} ms: '
        elif dur > 2e-6:
            msg = f' Time elapsed: {dur*1e6:.2f} us: '
        else:
            msg = f' Time elapsed: {dur:.2f} s: '

        self.tic()
        return msg