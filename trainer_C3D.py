import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from create_dataset import *
from C3D_model import *
from tqdm import *

import os
from tensorboardX import SummaryWriter


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', type=str, help='dataset root')
    parser.add_argument('--max_len', default=128, type=int, help='maximum length of RNAs')
    parser.add_argument('--num_RNA', default=100, type=int, help='number of postive RNAs')
    parser.add_argument('--num_negatives', default=500, type=int, help='number of negative RNAs corresponding to a specific positive one')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate of optimizer')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay of optimizer')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--total_steps', default=50000, type=int, help='total number of training steps')
    parser.add_argument('--milestone1', default=30000, type=int, help='first milestone to reduce the learning rate')
    parser.add_argument('--milestone2', default=40000, type=int, help='second milestone to reduce the learning rate')
    parser.add_argument('--batch_norm', action='store_true', help='apply BatchNorm in C4D model')
    parser.add_argument('--save_freq', default=2000, type=int, help='frequency for taking snapshot of the model')
    opt = parser.parse_args()

    model = C3D(num_classes=2, batch_norm=opt.batch_norm).cuda()
    cudnn.benchmark = True

    optimizer = optim.Adam(
            [
                {'params': get_1x_lr_params(model)},
                {'params': get_10x_lr_params(model),
                 'lr': opt.learning_rate * 10}
            ],
            lr=opt.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.milestone1,opt.milestone2], gamma=0.1)

    RNA_trainset = RNADataset(root=opt.data_root, train=True, num_RNA=opt.num_RNA, num_negatives=opt.num_negatives, max_len=opt.max_len, data_len=opt.total_steps)
    RNA_trainloader = torch.utils.data.DataLoader(dataset=RNA_trainset, batch_size=opt.batch_size)
    RNA_trainiter = iter(RNA_trainloader)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    writer = SummaryWriter(log_dir='tensorboard_logdir')
    for step in tqdm(range(opt.total_steps)):
        RNA_pos, RNA_neg, lbl_pos, lbl_neg, index = RNA_trainiter.__next__()
        input = torch.cat((RNA_pos, RNA_neg), dim=0).cuda()
        label = torch.cat((lbl_pos, lbl_neg), dim=0).view(-1).cuda()

        loss = model(input, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_losses = {'loss': loss}
        log_losses_tensorboard(writer, current_losses, step)

        if step % opt.save_freq == 0 and step != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),
                       os.path.join('checkpoints', f'model_{step}.pth'))
