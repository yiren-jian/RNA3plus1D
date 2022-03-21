import torch
import torch.nn as nn

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, batch_norm=False):
        super(C3D, self).__init__()
        self.batch_norm = batch_norm

        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True )
        self.bn1a = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True )
        self.bn2a = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True )
        self.bn3a = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True )
        self.bn4a = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()


    def forward(self, x, y=None):
        bsz = x.shape[0]
        length = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4, 5)
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.shape[5])

        x = self.conv1a(x)   ### conv1
        if self.batch_norm:
            x = self.bn1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)   ### conv2
        if self.batch_norm:
            x = self.bn2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)   ### conv3
        if self.batch_norm:
            x = self.bn3a(x)
        x = self.pool3(x)

        x = self.conv4a(x)   ### conv4
        if self.batch_norm:
            x = self.bn4a(x)
        x = self.pool4(x)

        x = x.view(x.shape[0], -1)
        logits = self.fc(x)

        if y is None:
            logits = logits.reshape(bsz, length, -1)
            probs = nn.Softmax(dim=-1)(logits)
            return probs.mean(dim=1)
        else:
            y = torch.stack([y  for i in range(length)]).T.reshape(-1)
            loss = nn.CrossEntropyLoss()(logits, y)
            return loss


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
         model.bn1a, model.bn2a, model.bn3a, model.bn4a, model.bn5a]
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
    inputs = torch.rand(6, 3, 128, 32, 32, 32).cuda()    #### bsz=1, one positive and one negative
    y = torch.tensor([0,0,0,1,1,1]).cuda()     #### label, 0 and 1 for two RNAs (positive and negative).
    net = C3D(num_classes=2, batch_norm=True).cuda()
    print('  Total params in share model: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    outputs = net.forward(inputs)
    # print(outputs.shape)
    print(outputs)
