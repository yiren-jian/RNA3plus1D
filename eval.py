import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from collections import OrderedDict

from create_dataset import *
from C4D_model import *
from tqdm import *

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', type=str, help='dataset root')
    parser.add_argument('--model_path', default='checkpoints/model_48000.pth', type=str, help='path to the trained model that needs to be evaluated')
    parser.add_argument('--max_len', default=128, type=int, help='maximum length of RNAs')
    parser.add_argument('--num_RNA', default=100, type=int, help='number of postive RNAs')
    parser.add_argument('--num_negatives', default=500, type=int, help='number of negative RNAs corresponding to a specific positive one')
    parser.add_argument('--batch_norm', action='store_true', help='apply BatchNorm in C4D model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')

    parser.add_argument('--eval_steps', default=5000, type=int, help='total number of evaluation steps (evaluated pairs)')
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

    #### the evaluation dataset, dataloader annd dataiter should be in the other form
    #### but let's adopt the trianing ones here at this moment
    #### will update later
    RNA_set = RNADataset(root=opt.data_root, train=True, num_RNA=opt.num_RNA, num_negatives=opt.num_negatives, max_len=opt.max_len, data_len=opt.eval_steps)
    RNA_loader = torch.utils.data.DataLoader(dataset=RNA_set, batch_size=opt.batch_size)
    RNA_iter = iter(RNA_loader)

    model.eval()    ### set model in evaluation mode: for testing behavior of Dropout and BatchNorm
    correct = 0.0    ### count number of corrected predicted pairs
    total = 0.0     #### total number of pairs being evaluated
    for step in tqdm(range(opt.eval_steps)):
        RNA_pos, RNA_neg, lbl_pos, lbl_neg, index = RNA_iter.__next__()    #### get data
        input = torch.cat((RNA_pos, RNA_neg), dim=0).cuda()    ### a mini-batch with a positive pair and a negative pair (here is 2=1+1)
        label = torch.cat((lbl_pos, lbl_neg), dim=0).view(-1).cuda()    #### concatenate the labels accordingly

        with torch.no_grad():
            output = model(input)    ### compute output for the mini-batch  ### output before softmax
            _, predicted = torch.max(output.data, 1)     #### take the predicted label, e.g. (0.11, 0.70, 0.19) is predicted to be 1, (0.11, 0.19, 0.70) is predicted to be 2
            total += label.size(0)    ### label.size(0) is batch size, here is 2=1+1
            correct += (predicted == label).sum().item()    #### (predicted == label).sum() calculates the number of corrected predicted

    print("Finally accuracy is %.4f"%(1 - correct/total))
