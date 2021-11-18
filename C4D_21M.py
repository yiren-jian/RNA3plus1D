import torch
import torch.nn as nn
from Conv4d import Conv4d

class C4D(nn.Module):
    """
    The C4D network.
    """

    def __init__(self, num_classes, batch_norm=False):
        super(C4D, self).__init__()
        self.batch_norm = batch_norm

        self.conv1a = Conv4d(3, 32, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn1a = nn.BatchNorm1d(32)
        self.conv1b = Conv4d(32, 32, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(2, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn1b = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2a = Conv4d(32, 64, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn2a = nn.BatchNorm1d(64)
        self.conv2b = Conv4d(64, 64, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(2, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn2b = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = Conv4d(64, 128, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn3a = nn.BatchNorm1d(128)
        self.conv3b = Conv4d(128, 128, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(2, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn3b = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = Conv4d(128, 256, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn4a = nn.BatchNorm1d(256)
        self.conv4b = Conv4d(256, 256, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(2, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn4b = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = Conv4d(256, 256, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn5a = nn.BatchNorm1d(256)
        self.conv5b = Conv4d(256, 256, kernel_size=(3, 3, 3, 3), padding=(1, 1, 1, 1), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True )
        self.bn5b = nn.BatchNorm1d(256)
        self.pool5 = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()


    def forward(self, x):

        x = self.conv1a(x)   ### conv
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn1a(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        x = self.conv1b(x)   ### conv
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn1b(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        N, C, L, D, H, W = x.size()
        x = self.pool1(x.view(-1, D, H, W))    ### 3D pool
        _, D, H, W = x.size()
        x = x.view(N, C, L, D, H, W)  ### L/2

        x = self.conv2a(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn2a(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        x = self.conv2b(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn2b(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        N, C, L, D, H, W = x.size()
        x = self.pool2(x.view(-1, D, H, W))
        _, D, H, W = x.size()
        x = x.view(N, C, L, D, H, W)  ### L/4

        x = self.conv3a(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn3a(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        x = self.conv3b(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn3b(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        N, C, L, D, H, W = x.size()
        x = self.pool3(x.view(-1, D, H, W))
        _, D, H, W = x.size()
        x = x.view(N, C, L, D, H, W)  ### L/8

        x = self.conv4a(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn4a(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        x = self.conv4b(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn4b(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        N, C, L, D, H, W = x.size()
        x = self.pool4(x.view(-1, D, H, W))
        _, D, H, W = x.size()
        x = x.view(N, C, L, D, H, W)  ### L/16

        x = self.conv5a(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn5a(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        x = self.conv5b(x)
        if self.batch_norm:
            N, C, L, D, H, W = x.size()
            x = x.view(N, C, -1)
            x = self.bn5b(x)    ### bn
            x = x.view(N, C, L, D, H, W)
        x = self.relu(x)    ### relu
        N, C, L, D, H, W = x.size()
        x = self.pool5(x.view(-1, D, H, W))
        _, D, H, W = x.size()
        x = x.view(N, C, L, D, H, W)  ### L/32

        ####
        x = x.view(-1, 2048)

        logits = self.fc(x)

        return logits


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1a, model.conv2a, model.conv3a, model.conv4a, model.conv5a,
         model.bn1a, model.bn2a, model.bn3a, model.bn4a, model.bn5a,
         model.conv1b, model.conv2b, model.conv3b, model.conv4b, model.conv5b,
         model.bn1b, model.bn2b, model.bn3b, model.bn4b, model.bn5b]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(4, 3, 128, 32, 32, 32).cuda()
    net = C4D(num_classes=2, batch_norm=True).cuda()
    print('  Total params in share model: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    outputs = net.forward(inputs)
    print(outputs.shape)
