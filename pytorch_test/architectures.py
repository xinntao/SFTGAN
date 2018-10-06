'''
Architectures for segmentation and SFTGAN models
'''
import torch.nn as nn
import torch.nn.functional as F
import block as B

####################
# Segmentation
####################


class Res131(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, dilation=1, stride=1):
        super(Res131, self).__init__()
        conv0 = B.conv_block(in_nc, mid_nc, 1, 1, 1, 1, False, 'zero', 'batch')
        conv1 = B.conv_block(mid_nc, mid_nc, 3, stride, dilation, 1, False, 'zero', 'batch')
        conv2 = B.conv_block(mid_nc, out_nc, 1, 1, 1, 1, False, 'zero', 'batch', None)  #  No ReLU
        self.res = B.sequential(conv0, conv1, conv2)
        if in_nc == out_nc:
            self.has_proj = False
        else:
            self.has_proj = True
            self.proj = B.conv_block(in_nc, out_nc, 1, stride, 1, 1, False, 'zero', 'batch', None)
            #  No ReLU

    def forward(self, x):
        res = self.res(x)
        if self.has_proj:
            x = self.proj(x)
        return nn.functional.relu(x + res, inplace=True)


class OutdoorSceneSeg(nn.Module):
    def __init__(self):
        super(OutdoorSceneSeg, self).__init__()
        # conv1
        blocks = []
        conv1_1 = B.conv_block(3, 64, 3, 2, 1, 1, False, 'zero', 'batch')  #  /2
        conv1_2 = B.conv_block(64, 64, 3, 1, 1, 1, False, 'zero', 'batch')
        conv1_3 = B.conv_block(64, 128, 3, 1, 1, 1, False, 'zero', 'batch')
        max_pool = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)  #  /2
        blocks = [conv1_1, conv1_2, conv1_3, max_pool]
        # conv2, 3 blocks
        blocks.append(Res131(128, 64, 256))
        for i in range(2):
            blocks.append(Res131(256, 64, 256))
        # conv3, 4 blocks
        blocks.append(Res131(256, 128, 512, 1, 2))  #  /2
        for i in range(3):
            blocks.append(Res131(512, 128, 512))
        # conv4, 23 blocks
        blocks.append(Res131(512, 256, 1024, 2))
        for i in range(22):
            blocks.append(Res131(1024, 256, 1024, 2))
        # conv5
        blocks.append(Res131(1024, 512, 2048, 4))
        blocks.append(Res131(2048, 512, 2048, 4))
        blocks.append(Res131(2048, 512, 2048, 4))
        blocks.append(B.conv_block(2048, 512, 3, 1, 1, 1, False, 'zero', 'batch'))
        blocks.append(nn.Dropout(0.1))
        # # conv6
        blocks.append(nn.Conv2d(512, 8, 1, 1))

        self.feature = B.sequential(*blocks)
        # deconv
        self.deconv = nn.ConvTranspose2d(8, 8, 16, 8, 4, 0, 8, False, 1)
        # softmax
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.feature(x)
        x = self.deconv(x)
        x = self.softmax(x)
        return x


#############################
# SFTGAN (pytorch version)
#############################


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1))

        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out


#############################################
# SFRGAN (torch version as in the paper)
#############################################


class SFTLayer_torch(nn.Module):
    def __init__(self):
        super(SFTLayer_torch, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift


class ResBlock_SFT_torch(nn.Module):
    def __init__(self):
        super(ResBlock_SFT_torch, self).__init__()
        self.sft0 = SFTLayer_torch()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer_torch()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv0(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFT_Net_torch(nn.Module):
    def __init__(self):
        super(SFT_Net_torch, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT_torch())
        sft_branch.append(SFTLayer_torch())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1))

        # Condtion network
        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(128, 32, 1))

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out
