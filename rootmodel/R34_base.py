import torch
import torch.nn as nn
import torch.nn.functional as F
from rootmodel.model_util import *
from rootmodel.model_base import *

class JointAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(JointAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_convFC = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())
        #
        self.sum_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())

    def forward(self, low, high):
        low_fea = self.avg_convFC(self.avg_pool(low)+self.max_pool(low))

        high_fea = self.max_convFC(self.max_pool(high)+self.avg_pool(high))

        sum_fea = low_fea + high_fea
        sum_fea = self.sum_convFC(sum_fea)

        low_out = low * sum_fea + low
        high_out = high * sum_fea + high
        return low_out, high_out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.low_con1 = ResNet34_1()
        self.low_con2 = ResNet34_2()
        self.low_con3 = ResNet34_3()
        self.low_con4 = ResNet34_4()
        self.low_con5 = ResNet34_5()

        self.high_con1 = ResNet34_1()
        self.high_con2 = ResNet34_2()
        self.high_con3 = ResNet34_3()
        self.high_con4 = ResNet34_4()
        self.high_con5 = ResNet34_5()

        self.JA1 = JointAttention(in_channel=64, ratio=16)
        self.JA2 = JointAttention(in_channel=64, ratio=16)
        self.JA3 = JointAttention(in_channel=128, ratio=16)
        self.JA4 = JointAttention(in_channel=256, ratio=16)
        self.JA5 = JointAttention(in_channel=512, ratio=16)
    def forward(self, low, high):
        low_1 = self.low_con1(low)
        high_1 = self.high_con1(high)
        low_1, high_1 = self.JA1(low_1, high_1)

        low_2 = self.low_con2(low_1)
        high_2 = self.high_con2(high_1)
        low_2, high_2 = self.JA2(low_2, high_2)

        low_3 = self.low_con3(low_2)
        high_3 = self.high_con3(high_2)
        low_3, high_3 = self.JA3(low_3, high_3)

        low_4 = self.low_con4(low_3)
        high_4 = self.high_con4(high_3)
        low_4, high_4 = self.JA4(low_4, high_4)

        low_5 = self.low_con5(low_4)
        high_5 = self.high_con5(high_4)
        low_5, high_5 = self.JA5(low_5, high_5)


        ## remember 3,4,5
        return low_3, low_4, low_5, high_3, high_4, high_5



class YHRSOD(nn.Module):
    def __init__(self):
        super(YHRSOD, self).__init__()
        self.Encoder = Encoder()

    def forward(self, low, high):
        low_3, low_4, low_5, high_3, high_4, high_5 = self.Encoder(low, high)
        # print("pre:")
        # print(low_3.size())
        # print(low_4.size())
        # print(low_5.size())
        # print(high_3.size())
        # print(high_4.size())
        # print(high_5.size())
        # pre:
        # innput: 4, 3, 256, 256
        # torch.Size([4, 128, 32, 32])
        # torch.Size([4, 256, 16, 16])
        # torch.Size([4, 512, 8, 8])
        # input: 4, 3, 256, 256
        # torch.Size([4, 128, 32, 32])
        # torch.Size([4, 256, 16, 16])
        # torch.Size([4, 512, 8, 8])
        return low, high