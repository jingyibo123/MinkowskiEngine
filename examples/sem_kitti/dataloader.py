
import numpy as np

import torch
import MinkowskiEngine as ME

from .dataset import SemanticKitti

from torch.utils.data import  DataLoader


class SemKittiDataLoader(DataLoader):

    class ToSparse(object):

        def __init__(self, voxel_size, max_count):
            self.voxel_size = voxel_size
            self.max_count = max_count
            pass

        def __call__(self, scan, label):
            """
            Args:


            Returns:

            """

            coords = scan[:, 0:3]
            int_coords = np.floor(coords / self.voxel_size)
            feats = scan[:, 3:]

            ind, ind_inv = ME.utils.sparse_quantize(int_coords,
                                                    ignore_label=-100,
                                                    return_index=True,
                                                    return_inverse=True)

            # print(f'Raw nb points: {feats.size}, voxelization: {self.voxel_size} : nb points: {ind.size}')
            if ind.size > self.max_count:
                print('[WARNING]: over max_count, subsampling...')
                # shuttle then subsample
                np.random.shuffle(ind)
                ind = ind[0:self.max_count]

                # subs = np.random.randint(0, ind.size, self.max_count, replace=False)
                # print(ind)
                # print(subs)
                # ind = ind[subs]
                # print(ind)
                # TODO correct??
                ind_inv = ind_inv[0:self.max_count]


            quantized_coords = int_coords[ind]
            quantized_feats = feats[ind]
            quantized_labels = label[ind]

            # return ME.SparseTensor(
            #     feats=torch.from_numpy(scan[:, 3:]),
            #     coords=ME.utils.batched_coordinates([scan[:, 0:3] / voxel_size]),
            #     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
            # )

            return (torch.IntTensor(quantized_coords), torch.FloatTensor(quantized_feats)), torch.IntTensor(quantized_labels)

        def __repr__(self):
            return self.__class__.__name__

    @staticmethod
    def collation(data):

        # flatten [((coords, feats), labels)] to [(coords, feats, labels)]
        data = [(item[0][0], item[0][1], item[1]) for item in data]

        return ME.utils.batch_sparse_collate(data),

    # Warning: read using mutable obects for default input arguments in python.
    def __init__(self, sequences_dir, data_yaml, split, voxel_size, max_count, **kwargs):


        super(SemKittiDataLoader, self).__init__(
            SemanticKitti(sequences_dir=sequences_dir, data_yaml=data_yaml, split=split,
                           transforms=self.ToSparse(voxel_size, max_count)
                         ),
            shuffle=True,
            collate_fn=self.collation,
            **kwargs
        )