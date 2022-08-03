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

class Decoder_A(nn.Module):
    def __init__(self):
        super(Decoder_A, self).__init__()
        self.low_con3 = DeResNet34_3()
        self.low_con4 = DeResNet34_4()
        self.low_con5 = DeResNet34_5()

        self.high_con3 = DeResNet34_3()
        self.high_con4 = DeResNet34_4()
        self.high_con5 = DeResNet34_5()

        self.lowEn3 = RFB(128, 128)
        self.lowEn4 = RFB(256, 256)
        self.lowEn5 = RFB(512, 512)
        self.highEn3 = RFB(128, 128)
        self.highEn4 = RFB(256, 256)
        self.highEn5 = RFB(512, 512)

    def forward(self, low_3, low_4, low_5, high_3, high_4, high_5):

        low_conv5_out = self.low_con5(self.lowEn5(low_5))
        low_conv4_out = self.low_con4(self.lowEn4(low_conv5_out + low_4))
        low_conv3_out = self.low_con3(self.lowEn3(low_conv4_out + low_3))

        high_conv5_out = self.high_con5(self.highEn5(high_5))
        high_conv4_out = self.high_con4(self.highEn4(high_conv5_out + high_4))
        high_conv3_out = self.high_con3(self.highEn3(high_conv4_out + high_3))

        # low: torch.Size([4, 64, 64, 64])
        # high: torch.Size([4, 64, 128, 128])
        return low_conv3_out, high_conv3_out

class PixUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PixUpBlock, self).__init__()
        self.up = nn.PixelShuffle(2)
        self.conv = CSBasicBlock(in_channel//4, out_channel, downsample=nn.Conv2d(in_channel//4, out_channel, 1, 1, 0))
    def forward(self, x):
        out = self.conv(self.up(x))
        return out

class UpDecoder(nn.Module):
    def __init__(self):
        super(UpDecoder, self).__init__()
        self.up1 = PixUpBlock(64, 32)
        self.up2 = PixUpBlock(32, 16)
    def forward(self, x):
        # B, 64, 64, 64
        out = self.up1(x)
        out = self.up2(out)
        return out

class YHRSOD(nn.Module):
    def __init__(self,
                 EnhanceCA_num=5,
                 DecoderB_num=5):
        super(YHRSOD, self).__init__()
        self.Encoder = Encoder()
        self.Decoder_A = Decoder_A()
        self.Enhancelow = make_layer(CSBasicBlock, EnhanceCA_num, inplanes=64, planes=64)
        self.Enhancehigh = make_layer(CSBasicBlock, EnhanceCA_num, inplanes=64, planes=64)
        self.Decoder_B = make_layer(CSBasicBlock, DecoderB_num, inplanes=64, planes=64)
        self.UpDecoder = UpDecoder()
        self.lastConv = nn.Conv2d(16, 1, 3, 1, 1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, low, high):
        low_3, low_4, low_5, high_3, high_4, high_5 = self.Encoder(low, high)
        low_deA, high_deA = self.Decoder_A(low_3, low_4, low_5, high_3, high_4, high_5)
        low_deA = self.Enhancelow(low_deA)
        high_deA = self.Enhancehigh(high_deA)
        high_deA = high_deA + self.upsample(low_deA)
        # B, 64, 128, 128
        out_decoder_B = self.Decoder_B(high_deA)
        out = self.UpDecoder(out_decoder_B)
        out = self.lastConv(out)

        return out


if __name__ == "__main__":
    model = YHRSOD()
    inputA = torch.randn(16, 3, 256, 256)
    inputB = torch.randn(16, 3, 512, 512)
    out = model(inputA, inputB)
    print(out.size())