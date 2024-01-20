import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.resnet import resnet50
from nets.vgg import VGG16


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat1 = nn.Sequential(                                        # 6、12相拼接
            nn.Conv2d(dim_out * 2, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat2 = nn.Sequential(                                       # 6、12、18相拼接
            nn.Conv2d(dim_out * 3, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)

        # 6、12相拼接
        x1 = torch.cat([x, conv3x3_1], dim=1)
        x1 = self.conv_cat1(x1)
        conv3x3_2 = self.branch3(x1)

        # 6、12、18相拼接
        x2 = torch.cat([x, conv3x3_1, conv3x3_2], dim=1)
        x2 = self.conv_cat2(x2)
        conv3x3_3 = self.branch4(x2)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        # 输入批次只有一个数据点，而由于BatchNorm操作必须要多于一个数据去计算平均值，如把batch_size的值改为大于1的数
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class up_conv(nn.Module):                              # AG之前进行升维
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv1(nn.Module):                                   # 最后融合是进行升维16、8、4、2
    def __init__(self,ch_in, ch_out, scale_factor):
        super(up_conv1,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(ch_in,ch_out, kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_final(nn.Module):                             # 最后进行融合
    def __init__(self,ch_in, ch_out):
        super(conv_final,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out, kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):                       # AG注意力机制
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # 1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        x1 = self.W_x(x)  # 1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        psi = self.relu(g1 + x1)  # 1x256x64x64di
        psi = self.psi(psi)  # 得到权重矩阵  1x256x64x64 -> 1x1x64x64 ->sigmoid 结果到（0，1）

        return x * psi  # 与low-level feature相乘，将权重矩阵赋值进去


class unetUp(nn.Module):                              # 原始两个拼接，其中一个要升维
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class con(nn.Module):                              # DASPP之后两个进行拼接
    def __init__(self, in_size, out_size):
        super(con, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]       # 表示拼接之后的特征维度数，如512+512=1024，512+256=768
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]   # 2048+1024   512+512
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        if backbone == 'vgg':
            # ASPP
            self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
            #
            self.Up4 = up_conv(ch_in=512, ch_out=512)
            self.Att4 = Attention_block(F_g=512, F_l=512, F_int=256)
            self.Up3 = up_conv(ch_in=512, ch_out=256)
            self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
            self.Up2 = up_conv(ch_in=256, ch_out=128)
            self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
            self.Up1 = up_conv(ch_in=128, ch_out=64)
            self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)

            self.aspp = ASPP(dim_in=512, dim_out=512, rate=1)
            self.concat = con(in_size=1024, out_size=512)

            self.up__4 = up_conv1(ch_in=512, ch_out=64, scale_factor=16)
            self.up__3 = up_conv1(ch_in=512, ch_out=64, scale_factor=8)
            self.up__2 = up_conv1(ch_in=256, ch_out=64, scale_factor=4)
            self.up__1 = up_conv1(ch_in=128, ch_out=64, scale_factor=2)

            self.final_ronghe = conv_final(ch_in=384, ch_out=64)

        elif backbone == "resnet50":
            self.Up4 = up_conv(ch_in=2048, ch_out=1024)
            self.Att4 = Attention_block(F_g=1024, F_l=1024, F_int=512)
            self.Up3 = up_conv(ch_in=512, ch_out=512)
            self.Att3 = Attention_block(F_g=512, F_l=512, F_int=256)
            self.Up2 = up_conv(ch_in=256, ch_out=256)
            self.Att2 = Attention_block(F_g=256, F_l=256, F_int=128)
            self.Up1 = up_conv(ch_in=128, ch_out=64)
            self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
            # 太大跑不了ASPP
            # self.aspp = ASPP(dim_in=2048, dim_out=2048, rate=1)
            # self.concat = con(in_size=4096, out_size=2048)

        #  upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # 如果backbone == 'resnet50'，那么在最后需要在进行一次上采样，因为resnet在最开始使用了7*7的卷积，使维度缩小了一倍512-256
        # 之后使用maxpooling进行下采样，维度缩小一倍256-128
        # 在第一个block没有下采样，在第二个block下采样，128-64，在第三个block下采样，64-32
        # 在第四个block下采样，32-16
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # # feat5输入到DASPP，然后在于feat5拼接，形成新的feat5
        # if self.backbone == "vgg":
        #     x = self.aspp(feat5)
        #     feat5 = self.concat(x, feat5)

        # feat5输入到DASPP到得x，然后x与feat5输入AG注意力机制y，x与y拼接得到新的feat5
        # if self.backbone == "vgg":
        #     x = self.aspp(feat5)
        #     y = self.Att5(x, feat5)
        #     feat5 = self.concat(x, y)
        x = self.aspp(feat5)               # 32*32*512
        y = self.Att5(x, feat5)
        feat5 = self.concat(x, y)             # 32*32*512

        # 先升维，然后AG注意力机制，最后拼接
        x4 = self.Up4(feat5)
        feat4 = self.Att4(g=x4, x=feat4)
        up4 = self.up_concat4(feat4, feat5)       # 64*64*512
        # print(up4.shape)

        # 先升维，然后AG注意力机制，最后拼接
        x3 = self.Up3(up4)
        # print(x3.shape)
        feat3 = self.Att3(g=x3, x=feat3)
        up3 = self.up_concat3(feat3, up4)       # 128*128*256

        # 先升维，然后AG注意力机制，最后拼接
        x2 = self.Up2(up3)
        feat2 = self.Att2(g=x2, x=feat2)
        up2 = self.up_concat2(feat2, up3)         # 256*256*128

        # 先升维，然后AG注意力机制，最后拼接
        x1 = self.Up1(up2)
        feat1 = self.Att1(g=x1, x=feat1)
        up1 = self.up_concat1(feat1, up2)         # 512*512*64

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        # final = self.final(up1)

        x = self.up__4(x)
        feat5 = self.up__4(feat5)
        up4 = self.up__3(up4)
        up3 = self.up__2(up3)
        up2 = self.up__1(up2)

        feature_cat = torch.cat([x, feat5, up4, up3, up2, up1], dim=1)
        feature = self.final_ronghe(feature_cat)

        final = self.final(feature)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
