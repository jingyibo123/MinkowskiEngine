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
import sys
import yaml

import sys
from os import path


import numpy as np

import torch
import MinkowskiEngine as ME

import torch.nn.parallel as parallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import torch.optim as optim

from examples.minkunet import *
from shutil import copyfile


from torchvision.datasets import VisionDataset

from examples.sem_kitti.tictoc import *

from examples.sem_kitti.dataloader import SemKittiDataLoader

from statistics import mean
import time


def getLatestCheckpoint(save_dir):
    import re
    latest = ('', 0, 0)
    for file in os.listdir(save_dir):
        m = re.search(r'.*_epoch_(\d*?)_batch_(\d*).*', file)
        if m:
            print('Found checkpoint:' + file)
            epoch = int(m.group(1))
            batch = int(m.group(2))
            if epoch >= latest[1] and batch >= latest[2]:
                latest = (file, epoch, batch)

    if latest[0] == '':
        return None
    else:
        return os.path.join(save_dir, latest[0]), latest[1], latest[2]




class Trainer(object):

    def __init__(self, data_yaml, config_yaml, sequences_dir, save_dir):
        """ Output Dir"""
        self.save_dir = save_dir
        self.resume_training = False
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            copyfile(data_yaml, save_dir+'/data.yaml')
            copyfile(config_yaml, save_dir+'/config.yaml')
        else:
            latest_checkpoint = getLatestCheckpoint(save_dir)
            if latest_checkpoint:
                self.resume_training = True

        import yaml

        """ Parse yaml files """
        self.data = yaml.safe_load(open(data_yaml, 'r', encoding="utf-8"))

        self.config = yaml.safe_load(open(config_yaml, 'r', encoding="utf-8"))

        """ Multiple GPUs & dataparallel """
        torch.cuda.manual_seed(0)

        self.devices = self.config['train']['devices']
        num_devices = torch.cuda.device_count()
        for device in self.devices:
            assert (device >= 0 and device < num_devices)

        self.target_device = self.config['train']['target_device']
        assert (self.target_device >= 0 and self.target_device < num_devices and self.target_device in self.devices)
        self.multi_gpus = len(self.devices) != 1

        self.train_batch_size = len(self.devices) * self.config['train']['batch_size']
        print('Using ', len(self.devices), ' GPUs: ' + str(self.devices) + '. Total batch size: ',
              self.train_batch_size)
        self.valid_batch_size = len(self.devices) * self.config['valid']['batch_size']

        os.environ['OMP_NUM_THREADS'] = str(self.config['train']['omp_num_threads'])

        """ Input """
        self.sequences_dir = sequences_dir
        self.train_loader = SemKittiDataLoader(sequences_dir=sequences_dir,
                                               data_yaml=self.data,
                                               split='train',
                                               batch_size=self.config['train']['batch_size'],
                                               voxel_size=self.config['preprocessing']['voxel_size'],
                                               max_count=self.config['preprocessing']['max_count'],
                                               num_workers=self.config['train']['workers'])

        self.val_loader = SemKittiDataLoader(sequences_dir=sequences_dir,
                                             data_yaml=self.data,
                                             split='valid',
                                             batch_size=self.config['valid']['batch_size'],
                                             voxel_size=self.config['preprocessing']['voxel_size'],
                                             max_count=self.config['preprocessing']['max_count'],
                                             num_workers=self.config['valid']['workers'])

        self.nclasses = len(self.data['learning_map_inv'])


        """ Network related"""
        # TODO dynamic import
        import importlib
        self.net = MinkUNet14A(self.config['network']['feature_dimension'], self.nclasses, D=self.config['network']['dimension'])

        if self.resume_training:
            print('Loading network from '+latest_checkpoint[0])
            self.net.load_state_dict(torch.load(latest_checkpoint[0]))

        self.net.to(device=self.target_device)
        if self.multi_gpus:
            # Synchronized batch norm
            self.net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.net)

        if not torch.cuda.is_available():
            # TODO cpu traning
            pass



        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.config['train']['lr'],
            momentum=self.config['train']['momentum'],
            weight_decay=self.config['train']['w_decay'])

        # Copy the loss layer
        content = torch.zeros(len(self.data['learning_map_inv']), dtype=torch.float32)
        for cl, freq in self.data["content"].items():
            content[self.data['learning_map'][cl]] += freq
        loss_w = 1 / (content + self.config['train']['epsilon_w'])  # get weights
        for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
            if self.data["learning_ignore"][x_cl]:
                # don't weigh
                loss_w[x_cl] = 0

        print("Loss weights from content: ", loss_w.data)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=loss_w).to(self.target_device)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=-100).to(self.target_device)
        self.criterions = parallel.replicate(self.criterion, self.devices)


        """ Eval """
        from examples.sem_kitti.ioueval import iouEval
        self.evaluator = iouEval(self.nclasses, torch.device('cpu'), [])

        self.epoch = 0
        self.nb_batches = int(len(self.train_loader.dataset)) // (len(self.devices)*self.config['train']['batch_size'])
        self.batch = 0
        self.best_miou = -1.0

        if self.resume_training:
            self.epoch = latest_checkpoint[1]
            self.batch = latest_checkpoint[2]
            self.epoch += 1


    def train(self):

        if self.resume_training:
            self.eval()

        for epoch in range(self.epoch, self.config['train']['max_epochs']):
            self.epoch = epoch
            self.batch = 0

            self.train_epoch()

            self.eval()


    def train_epoch(self):
        # TEST
        # import pickle
        # # Get new data
        # it = iter(self.train_loader)
        # t = TicToc()
        # id = 0
        # end_batch = False
        # while not end_batch:
        #
        #     try:
        #         item = next(it)
        #     except StopIteration as e:
        #         end_batch = True
        #
        #     print(t.toc() + 'Read from loader')
        #     (coords, feats, labels) = item[0]
        #     tensor = ME.SparseTensor(feats, coords)
        #
        #     print(t.toc() + 'Construct sparse tensor')
        #
        #     with open('/media/yibo/DATA/tmp/KITTI/' + str(id) + '.pkl', 'wb') as f:
        #         torch.save(tensor, f)
        #         # pickle.dump(tensor, f)
        #     print(t.toc() + 'Dump to disk')
        # sys.exit(0)
        # # TEST

        self.net.train()

        times = []
        losses = []
        accs = []
        mious = []

        it = iter(self.train_loader)

        t = TicToc()

        end_batch = False
        while not end_batch:

            start_t = time.time()

            # Get new data
            inputs, all_labels = [], []
            for device in self.devices:

                t.tic()

                try:
                    item = next(it)
                except StopIteration as e:
                    end_batch = True

                (coords, feats, labels) = item[0]

                with torch.cuda.device(device):
                    inputs.append(
                        ME.SparseTensor(feats, coords).to(device))

                all_labels.append(labels.long().to(device))
                if end_batch:
                    break

            # print(t.toc() + 'gather data and send to GPU')
            # print('Size of input: ' + str(inputs[0].shape))

            using_devices = self.devices[0: len(inputs)]
            replicas = parallel.replicate(self.net, using_devices)
            outputs = parallel.parallel_apply(replicas, inputs, devices=using_devices)

            # print(t.toc() + ' forward')

            # Extract features from the sparse tensors to use a pytorch criterion
            out_features = [output.F for output in outputs]

            self.optimizer.zero_grad()

            all_losses = parallel.parallel_apply(
                self.criterions[0: len(inputs)],
                tuple(zip(out_features, all_labels)),
                devices=using_devices)

            # print(t.toc() + ' loss')

            loss = parallel.gather(all_losses, self.target_device, dim=0).mean()

            # print(t.toc() + ' gather loss')

            loss.backward()

            # print(t.toc() + ' backward loss')

            self.optimizer.step()

            # print(t.toc() + ' optimizer')

            for i in range(len(using_devices)):
                pred = outputs[i].F.max(1)[1]  # get the index of the max log-probability
                self.evaluator.addBatch(pred, all_labels[i])

            miou, _ = self.evaluator.getIoU()
            acc = self.evaluator.getacc()
            self.evaluator.reset()

            losses.append(loss.item())
            times.append(time.time() - start_t)
            accs.append(acc.item())
            mious.append(miou.item())

            if not self.batch % self.config['train']['report_batch']:
                print(
                      # f'Lr: {lr:.3e} | '
                      # f'Update: {umean:.3e} mean,{ustd:.3e} std | '
                      f'Epoch: [{self.epoch}][{self.batch}/{self.nb_batches}] | '
                      f'Time(s) {times[-1]:.3f} ({mean(times):.3f}) | '
                      f'Loss {losses[-1]:.4f} ({mean(losses):.4f}) | '
                      f'acc {accs[-1]:.3f} ({mean(accs):.3f}) | '
                      f'mIoU(%) {100 * mious[-1]:.1f} ({100 * mean(mious):.1f})'
                )

            if self.batch % 1000 == 0 and self.batch != 0:
                self.eval()


            self.batch += 1

    def eval(self):


        from examples.sem_kitti.ioueval import iouEval

        # make evaluator
        evaler = iouEval(self.nclasses, torch.device('cpu'), [])

        torch.cuda.empty_cache()

        self.net.eval()

        test_loss = 0
        correct = 0
        val_iter = 0
        total = 0

        print('Evaluating: ')

        it = iter(self.val_loader)
        end_batch = False
        while not end_batch:
            with torch.no_grad():
                # Get new data
                inputs, all_labels = [], []
                for device in self.devices:
                    try:
                        item = next(it)
                    except StopIteration as e:
                        end_batch = True

                    (coords, feats, labels) = item[0]

                    with torch.cuda.device(device):
                        inputs.append(
                            ME.SparseTensor(feats, coords).to(device))

                    all_labels.append(labels.long().to(device))
                    if end_batch:
                        break

                using_devices = self.devices[0: len(inputs)]
                replicas = parallel.replicate(self.net, using_devices)
                outputs = parallel.parallel_apply(replicas, inputs, devices=using_devices)

                # Extract features from the sparse tensors to use a pytorch criterion
                out_features = [output.F for output in outputs]
                losses = parallel.parallel_apply(
                    self.criterions[0: len(inputs)], tuple(zip(out_features, all_labels)), devices=using_devices)
                loss = parallel.gather(losses, self.target_device, dim=0).mean()

                test_loss += loss.item()

                for i in range(len(using_devices)):
                    pred = outputs[i].F.max(1)[1]  # get the index of the max log-probability

                    # run
                    evaler.addBatch(pred.cpu(), all_labels[i].cpu())

                val_iter = val_iter + 1

                if val_iter % 100 == 0:
                    print(
                        f'Evaluating: iter: {val_iter}, Loss: {test_loss / val_iter}'
                    )

        test_loss = test_loss / len(self.val_loader)  # average over number of mini-batch

        acc = evaler.getacc()

        m_iou, iou = evaler.getIoU()

        print('Validation set: Average loss: {:.4f}, Accuracy: ({:.2f}%), mIoU: ({:.2f}%)'.format(
            test_loss,  100 * acc, 100 * m_iou))


        for i, jacc in enumerate(iou):
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=self.data['labels'][self.data['learning_map_inv'][i]], jacc=jacc).encode('utf-8'))

        if m_iou > self.best_miou:
            # not saving for the first eval after resuming
            if not self.resume_training or self.best_miou > 0.0:
                print('Best mean IoU so far, saving model...')
                torch.save(self.net.state_dict(), self.save_dir + '/check_point_epoch_'+str(self.epoch) + '_batch_' + str(self.batch))

            self.best_miou = m_iou

        self.net.train()

        pass




