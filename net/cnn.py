import torch
import torch.nn as nn
from torch.nn.init import xavier_normal


def conv(batch_norm, in_channel, out_channel, k_size=3, stride=1):
    net = None
    if batch_norm:
        net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, k_size, stride,
                      padding=(k_size - 1)//2, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    else:
        net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, k_size, stride,
                      padding=(k_size - 1)//2, bias=True),
            nn.ReLU(),
        )
    return net


def fullc(in_channel, out_channel, activation=False):
    if activation:
        return nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(),
        )
    else:
        return nn.Linear(in_channel, out_channel)


class Net(nn.Module):
    def __init__(self, batch_norm=False):
        super(Net, self).__init__()
        self.batch_norm = batch_norm
        # O = (I-ks+2*P)/2 + 1 , P=(ks-1) // 2
        # 1280,384,6
        self.conv1 = conv(self.batch_norm, 6, 64, k_size=7, stride=2)
        # 640,192,64
        self.conv2 = conv(self.batch_norm, 64, 128, k_size=5, stride=2)
        # 320,96,128
        self.conv3 = conv(self.batch_norm, 128, 256, k_size=5, stride=2)
        # 160,48,256
        self.conv3_1 = conv(self.batch_norm, 256, 256)
        self.conv4 = conv(self.batch_norm, 256, 512, stride=2)
        # 80,32,512
        self.conv4_1 = conv(self.batch_norm, 512, 512)
        self.conv5 = conv(self.batch_norm, 512, 512, stride=2)
        # 40,12,512
        self.conv5_1 = conv(self.batch_norm, 512, 512)
        self.conv6 = conv(self.batch_norm, 512, 1024, stride=2)
        # 20,6,1024
        self.conv6_1 = conv(self.batch_norm, 1024, 1024)
        # 10,3,1024
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc_1 = fullc(1024 * 3 * 10, 4096, activation=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_2 = fullc(4096, 1024, activation=True)
        self.fc_3 = fullc(1024, 128, activation=True)
        self.fc_4 = fullc(128, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果m类型是nn.Conv2d，采用xavier_normal初始化W
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):  # 如果m类型是nn.BatchNorm2d，采用1填充W
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        def forward(self, x1, x2):
            x = torch.cat([x1, x2], dim=1)
            x = self.conv2(self.conv1(x))
            x = self.conv3_1(self.conv3(x))
            x = self.conv4_1(self.conv4(x))
            x = self.conv5_1(self.conv5(x))
            x = self.conv6_1(self.conv6(x))
            x = self.pool_1(x)
            x = x.view(x.size(0), -1)
            x = self.dropout1(x)
            x = self.fc_1(x)
            x = self.dropout2(x)
            x = self.fc_2(x)
            x = self.fc_3(x)
            x = self.fc_4(x)

            return x

        def weight_parameters(self):
            return [param for name, param in self.named_parameters() if 'weight' in name]

        def bias_parameters(self):
            return [param for name, param in self.named_parameters() if 'bias' in name]

def main():
    net  =Net()
    print(net)

if __name__ == '__main__':
    main()
    