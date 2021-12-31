import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from collections import OrderedDict

### use the dataset defined in this own file
from C4D_model import *
from tqdm import *

import os

from torch.utils.data.dataset import Dataset
import numpy as np


class RNADataset(Dataset):
    def __init__(self, root, max_len=128):
        self.root = os.path.expanduser(root)
        self.max_len = max_len

        # read the data file
        self.data_path = root + '/predict'
        self.RNAs = os.listdir(self.data_path)      ###### get a list with all files in this directory
        self.RNAs = [rna for rna in self.RNAs if rna.endswith('.npy')]      ####### picking only files ending with .npy

        # calculate data length
        self.data_len = len(self.RNAs)    #### length of dataset is number of RNAs in the folder

    def __getitem__(self, idx):
        RNA_name = self.RNAs[idx]
        RNA_tensor = torch.from_numpy(np.load(self.data_path + '/' + RNA_name)).permute(1,0,2,3,4)

        RNA_tensor = RNA_tensor.permute(0,4,2,3,1)
        l = RNA_tensor.size(-1)
        RNA_tensor = F.pad(RNA_tensor, (0, self.max_len-l), "constant", 0)
        RNA_tensor = RNA_tensor.permute(0,4,2,3,1)

        return RNA_tensor.float(), idx

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', type=str, help='dataset root')
    parser.add_argument('--model_path', default='checkpoints/model_48000.pth', type=str, help='path to the trained model that needs to be evaluated')
    parser.add_argument('--max_len', default=128, type=int, help='maximum length of RNAs')
    parser.add_argument('--batch_norm', action='store_true', help='apply BatchNorm in C4D model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    opt = parser.parse_args()

    #### initialize the model, the same architecture of the training one
    model = C4D(num_classes=2, batch_norm=opt.batch_norm).cuda()

    #### load the state_dict (dictionary for storing trained parameters)
    model_ckpt = torch.load(opt.model_path)    #### or it may be model_ckpt = torch.load(opt.model_path)['state_dict']
    ### the following is for in case the model being trained using nn.DataParellel
    new_state_dict = OrderedDict()
    for k, v in model_ckpt.items():
        name = k.replace("module.","")
        new_state_dict[name]=v
    ####### the above may not be necessary is the model is trained on a single GPU

    #### copy parameters back to model
    model.load_state_dict(new_state_dict, strict=True)

    #### dataset
    #### put all x_xxx.npy in data/predict
    RNA_set = RNADataset(root=opt.data_root,  max_len=opt.max_len)
    RNA_loader = torch.utils.data.DataLoader(dataset=RNA_set, batch_size=opt.batch_size, shuffle=False)

    model.eval()    ### set model in evaluation mode: for testing behavior of Dropout and BatchNorm
    RNAs_predictions = []
    RNAs_names = []
    for i, data in enumerate(RNA_loader):
        input, idx = data    #### get data
        input = input.cuda()

        with torch.no_grad():
            output = model(input)    ### compute output for the mini-batch  ### output before softmax
            predicted_prob = nn.Softmax(dim=1)(output)
            predicted_prob = predicted_prob.detach().data

            RNAs_names.append(RNA_set.RNAs[idx])   #### get the name ("0_100.npy") for this RNA
            RNAs_predictions.append(predicted_prob.cpu().numpy().tolist())    #### prediction

    for item in (zip(RNAs_names, RNAs_predictions)):
        print(item)    #### show the results
