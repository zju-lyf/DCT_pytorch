""" 
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from torchvision import datasets, models, transforms

__all__ = ['xception']

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}
transform = transforms.Compose([
        #transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_filters)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        x = self.ca(x)*x

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x
class Xception_1(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_1, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def xception_1(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception_1(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1_1 = Block(48, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # -----------------------------

    def forward(self, x):
        #(32*3*299*299)
        '''x = self.conv1(x)
        #torch.Size([32, 32, 149, 149])
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        #torch.Size([32, 64, 147, 147])
        x = self.bn2(x)
        x = self.relu(x)'''

        x = self.block1_1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #print (x.shape)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        #print (x.shape)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        #print (x.shape)
        x = self.fc1(x)

        return x


def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model


class XcepDCT_Upscaled_Static(nn.Module):
    def __init__(self, channels=0, pretrained=False, input_gate=False):
        super(XcepDCT_Upscaled_Static, self).__init__()

        self.input_gate = input_gate

        '''xception1 = xception_1(pretrained=True)
        class_num = 2
        channel_in = xception1.fc.in_features
        xception1.fc = nn.Linear(channel_in,class_num)
        model = xception(num_classes=2)
        pretrained_dict = xception1.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)'''

        #model = xception(num_classes=2, pretrained=pretrained)
        self.SELayer = SELayer(channels)
        self.ca = ChannelAttention(channels)
        model = xception(num_classes=2)

        #self.model = nn.Sequential(*list(model.children())[5:-1])
        self.model = model
        self.fc = list(model.children())[-1]
        self.relu = nn.ReLU(inplace=True)
        #print (self.model[0])

        if channels == 0 or channels == 192:
            out_ch = self.model[0][0].conv1.out_channels
            self.model[0][0].conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].conv1)

            out_ch = self.model[0][0].downsample[0].out_channels
            self.model[0][0].downsample[0] = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(self.model[0][0].downsample[0])

            # temp_layer = conv3x3(channels, out_ch, 1)
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].conv1.weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].conv1 = temp_layer

            # out_ch = self.model[0][0].downsample[0].out_channels
            # temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            # temp_layer.weight.data = self.model[0][0].downsample[0].weight.data.repeat(1, 3, 1, 1)
            # self.model[0][0].downsample[0] = temp_layer
        '''elif channels < 64:
            print (self.model[0])
            out_ch = self.model[0].rep[1].conv1.out_channels
            # temp_layer = conv3x3(channels, out_ch, 1)
            temp_layer = nn.Conv2d(channels, channels, kernel_size=3, stride=1, bias=False)
            temp_layer.weight.data = self.model[0].rep[1].conv1.weight.data[:, :channels]
            self.model[0].rep[1].conv1 = temp_layer

            out_ch = self.model[0].rep[4].conv1.out_channels
            temp_layer = nn.Conv2d(channels, out_ch, kernel_size=1, stride=1, bias=False)
            temp_layer.weight.data = self.model[0].rep[4].conv1.weight.data[:, :channels]
            self.model[0].rep[4].conv1 = temp_layer
            print (self.model[0])'''

        if input_gate:
            self.inp_GM = GateModule192()
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        if self.input_gate:
            x, inp_atten = self.inp_GM(x)
        #print (x.shape)
        x = self.ca(x)*x
        #print (x.shape)
        x = self.model(x)
        #print(x.shape)
        if self.input_gate:
            return x, inp_atten
        else:
            return x
class Xcep_ycbcr(nn.Module):
    def __init__(self, num_classes=2, pretrained = True, batch_size = 2, channels = 48):
        super(Xcep_ycbcr, self).__init__()
        self.channels = channels
        self.backbone = XcepDCT_Upscaled_Static(channels=channels)
        self.batch_size = batch_size
        #self.model = xception(num_classes=2)
    def forward(self,input):
        #out = self.model(input)
        (Y_dct,CR_dct,CB_dct) = input
        #print(len(Y_dct))
        for i in range(self.batch_size):
            y_dct_list = Y_dct[i]
            cr_dct_list = CR_dct[i]
            cb_dct_list = CB_dct[i]
            for j in range(len(y_dct_list)):
                y_dct_list[j] = transform(y_dct_list[j])
                y_dct_list[j] = y_dct_list[j].unsqueeze(0)
                cr_dct_list[j] = transform(cr_dct_list[j])
                cr_dct_list[j] = cr_dct_list[j].unsqueeze(0)
                cb_dct_list[j] = transform(cb_dct_list[j])
                cb_dct_list[j] = cb_dct_list[j].unsqueeze(0)
            for n in range(1, len(y_dct_list)):
                if n == 1:
                    Y_DCT = torch.cat([y_dct_list[0], y_dct_list[1]], dim=1)
                    CR_DCT = torch.cat([cr_dct_list[0], cr_dct_list[1]], dim=1)
                    CB_DCT = torch.cat([cb_dct_list[0], cb_dct_list[1]], dim=1)
                else:
                    Y_DCT = torch.cat([Y_DCT, y_dct_list[n]], dim=1)
                    CR_DCT = torch.cat([CR_DCT, cr_dct_list[n]], dim=1)
                    CB_DCT = torch.cat([CB_DCT, cb_dct_list[n]], dim=1)
            #print (Y_DCT.shape)
            if i == 0:
                input_dct = torch.cat((Y_DCT,CR_DCT,CB_DCT), dim = 1)
            else:
                temp = torch.cat((Y_DCT,CR_DCT,CB_DCT), dim = 1)
                input_dct = torch.cat((input_dct,temp), dim=0)
        #print (input_dct.shape)
        input_dct = input_dct.cuda()
        output = self.backbone(input_dct)
        return output