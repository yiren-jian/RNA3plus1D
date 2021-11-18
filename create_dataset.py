from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random


class RNADataset(Dataset):
    def __init__(self, root, train=True, num_RNA=100, num_negatives=500, max_len=512, data_len=50000):
        self.train = train
        self.root = os.path.expanduser(root)
        self.num_RNA = num_RNA
        self.num_negatives = num_negatives
        self.max_len = max_len

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = data_len

    def __getitem__(self, i):
        positive_index = random.choice(range(self.num_RNA))
        positive_RNA = torch.from_numpy(np.load(self.data_path + '/%d_0.npy'%(positive_index)))
        negative_index = random.choice(range(1, self.num_negatives+1))
        negative_RNA = torch.from_numpy(np.load(self.data_path + '/%d_%d.npy'%(positive_index, negative_index)))

        positive_RNA = positive_RNA.permute(0,4,2,3,1)
        l = positive_RNA.size(-1)
        positive_RNA = F.pad(positive_RNA, (0, self.max_len-l), "constant", 0)
        positive_RNA = positive_RNA.permute(0,4,2,3,1)

        negative_RNA = negative_RNA.permute(0,4,2,3,1)
        l = negative_RNA.size(-1)
        negative_RNA = F.pad(negative_RNA, (0, self.max_len-l), "constant", 0)
        negative_RNA = negative_RNA.permute(0,4,2,3,1)

        pos = torch.ones(1)
        neg = torch.zeros(1)

        return positive_RNA.float(), negative_RNA.float(), pos.long(), neg.long(), positive_index

    def __len__(self):
        return self.data_len
