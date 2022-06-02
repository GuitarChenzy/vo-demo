# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from models.resnet_encoder import *


def fc(c_in, c_out, activation=False):
    if activation:
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
        )
    else:
        return nn.Linear(c_in, c_out)


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.dropout1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=6 * 8 * 26, hidden_size=1024, num_layers=2, batch_first=True, dropout=0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_lstm_1 = fc(1024, 128, activation=True)
        self.fc_lstm_2 = fc(128, 6)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        print(out.size())
        x = out.view(1, -1, 6 * 8 * 26)
        print('x: ',x.size())
        x = self.dropout1(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        # print(x.size())
        x = x.contiguous().view(1, -1, 1024)
        x = self.fc_lstm_1(x)
        x = self.fc_lstm_2(x)
        print('x: ', x.size())

        out = x.mean(1)
        print('out: ', out.size())  # torch.Size([2, 6])
        pose = 0.01 * out.view(-1, 6)
        print('out: ', out.size())

        return pose


class PoseResNet(nn.Module):

    def __init__(self, num_layers=18, pretrained=True):
        super(PoseResNet, self).__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, imgL1, imgL2, imgR1, imgR2):
        xl = torch.cat([imgL1, imgL2], 1)
        xr = torch.cat([imgR1, imgR2], 1)
        print(xl.size())
        featuresl = self.encoder(xl)
        featuresr = self.encoder(xr)
        pose_l = self.decoder([featuresl])
        pose_r = self.decoder([featuresr])
        return pose_l, pose_r


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    model = PoseResNet().cuda()
    model.train()

    img_l1 = torch.randn(1, 4, 256, 832).cuda()
    img_l2 = torch.randn(1, 4, 256, 832).cuda()
    img_r1 = torch.randn(1, 4, 256, 832).cuda()
    img_r2 = torch.randn(1, 4, 256, 832).cuda()
    pose = model(img_l1, img_l2, img_r1, img_r2)

    print('pose: ', pose[0],'\n',pose[1])
    # from torchsummary import summary
    #
    # summary(model, img_l1, img_l2, img_r1, img_r2)
