
import os

import numpy as np

from torchvision.datasets import VisionDataset


class SemanticDataset:
    def __init__(self):
        # train, valid or test
        # split: str
        # nclasses: int
        # labels: dict
        # color_map: dict
        # learning_map: dict
        # learning_map_inv: dict
        # contents: dict


        self.split = ''

        self.nclasses = 0

        # Raw ID to label
        self.labels = []

        # Raw ID to color
        self.color_map = []

        # Raw ID to train ID
        self.learning_map = []

        # Train ID to raw ID
        self.learning_map_inv = []

        # Raw ID to percentage of class in train data
        self.contents = []



class SemanticKitti(VisionDataset, SemanticDataset):

  def __init__(self, sequences_dir,
               data_yaml, split, **kwargs):
    """

    :param sequences_dir:
    :param data_yaml:
    :param split: 'train', 'valid' or 'test'
    """
    super(SemanticKitti, self).__init__(sequences_dir, **kwargs)

    self.sequences_dir = sequences_dir
    self.data_yaml = data_yaml
    self.split = split

    self.sequences = self.data_yaml["split"][self.split]

    self.labels = self.data_yaml["labels"]
    self.color_map = self.data_yaml["color_map"]

    self.learning_map = self.data_yaml["learning_map"]
    self.remap_lut = np.zeros((500), dtype=np.int32)
    for key, data in self.learning_map.items():
        self.remap_lut[key] = data

    self.learning_map_inv = self.data_yaml["learning_map_inv"]

    self.content = self.data_yaml["content"]

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    self.remap_lut_inv = np.zeros((self.nclasses), dtype=np.int32)
    for key, data in self.learning_map_inv.items():
        self.remap_lut_inv[key] = data

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.sequences_dir):
      print("Sequences folder exists! Using sequences from %s" % self.sequences_dir)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete

    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))
      assert(os.path.exists(os.path.join(self.sequences_dir, seq)))

      # get paths for each
      scan_path = os.path.join(self.sequences_dir, seq, "velodyne")
      label_path = os.path.join(self.sequences_dir, seq, "labels")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn ]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn ]

      # check all scans have labels
      assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()


    # # FIXME
    # self.scan_files = self.scan_files[:20]
    # self.label_files = self.label_files[:20]

    print(f"Using {len(self.scan_files)} scans from {self.split} sequences {self.sequences}")

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]

    label_file = self.label_files[index]

    # open and obtain scan
    scan = np.fromfile(scan_file, dtype=np.float32)
    scan = scan.reshape(-1, 4)

    label = np.fromfile(label_file, dtype=np.uint32)

    label = label & 0xFFFF  # semantic label in lower half
    label = self.remap_lut[label]


    # sz = 535000
    # scan = np.zeros([sz, 4], dtype=np.float32)
    # for i in range(sz):
    #     scan[i, 0] = i
    #     scan[i, 1] = sz % 5
    #     scan[i, 3] = 0.2
    # label = np.ones([sz], dtype=np.int32)
    # for i in range(sz):
    #     label[i] = sz % 5

    if self.transforms is not None:
        scan, label = self.transforms(scan, label)


    return scan, label

  def __len__(self):
    return len(self.scan_files)
