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
import torch_dct as DCT
__all__ = ['xception']

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}
transform = transforms.Compose([
        transforms.Resize((299, 299)),
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

class Block_dct(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block_dct, self).__init__()

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

class Block_img(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block_img, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        #self.ca = ChannelAttention(out_filters)
        self.sa = SpatialAttention()
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
        x = self.sa(x)*x

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Xception_img(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, fusion='fc'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_img, self).__init__()
        self.num_classes = num_classes
        self.fusion = fusion

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block_img(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block_img(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block_img(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block_img(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block_img(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def forward(self, input):
        out = {}
        (input1, num) = input
        if num == 7:
            x = self.conv1(input1)
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
            out['block7'] = x
            return x
        elif num == 12:
            x = self.block8(input1)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
            out['block12'] = x
            return x
        elif num == 0:
            x = self.conv3(input1)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            out['final'] = x
            return x
def xception_img(num_classes=1000, pretrained=False, fusion='fc'):
    model = Xception_img(num_classes=num_classes, fusion=fusion)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model
class Xception_dct(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, fusion='fc'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_dct, self).__init__()
        self.num_classes = num_classes
        self.fusion = fusion

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block_dct(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block_dct(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block_dct(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block_dct(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block_dct(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def forward(self, input):
        out = {}
        (input1, num) = input
        if num == 7:
            x = self.conv1(input1)
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
            out['block7'] = x
            return x
        elif num == 12:
            x = self.block8(input1)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
            out['block12'] = x
            return x
        elif num == 0:
            x = self.conv3(input1)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.last_linear(x)
            out['final'] = x
            return x
def xception_dct(num_classes=1000, pretrained=False, fusion='fc'):
    model = Xception_dct(num_classes=num_classes, fusion=fusion)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model
class Xcep_ycbcr(nn.Module):
    def __init__(self, batchsize):
        super(Xcep_ycbcr, self).__init__()
        xception1 = xception_1(pretrained=True)
        model1 = xception_img(num_classes=2)
        pretrained_dict = xception1.state_dict()
        model1_dict = model1.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model1_dict}
        model1_dict.update(pretrained_dict)
        model1.load_state_dict(model1_dict)
        model2 = xception_dct(num_classes=2)
        pretrained_dict = xception1.state_dict()
        model2_dict = model2.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model2_dict}
        model2_dict.update(pretrained_dict)
        model2.load_state_dict(model2_dict)
        self.backbone_img = model1
        self.backbone_dct = model2
        self.post_function = nn.Softmax(dim=1)
        self.batchsize = batchsize
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        #self.model = xception(num_classes=2)
    def forward(self,input):
        #out = self.model(input)
        (image, Y_dct,CR_dct,CB_dct) = input
        for i in range(self.batchsize):
            y_dct = Y_dct[i]
            cr_dct = CR_dct[i]
            cb_dct = CB_dct[i]
            y_dct = transform(y_dct)
            #print (y_dct.shape)
            y_dct = y_dct.unsqueeze(0)
            cr_dct = transform(cr_dct)
            cr_dct = cr_dct.unsqueeze(0)
            cb_dct = transform(cb_dct)
            cb_dct = cb_dct.unsqueeze(0)

            '''for j in range(len(y_dct_list)):
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
                    CB_DCT = torch.cat([CB_DCT, cb_dct_list[n]], dim=1)'''
            #print (Y_DCT.shape)
            if i == 0:
                input_dct = torch.cat((y_dct,cr_dct,cb_dct), dim = 1)
                #print (input_dct.shape)
            else:
                temp = torch.cat((y_dct,cr_dct,cb_dct), dim = 1)
                input_dct = torch.cat((input_dct,temp), dim=0)
        #print (input_dct.shape)
        dct = input_dct.cuda()
        img_7_input = (image, 7)
        img_7 = self.backbone_img(img_7_input)
        dct_7_input = (dct, 7)
        dct_7 = self.backbone_dct(dct_7_input)
        img_7_sa = self.sa1(img_7)
        dct_7_sa = self.sa2(dct_7)
        # print (dct_7.shape)
        idct_7 = DCT.idct(dct_7)
        dct_img_7 = DCT.dct(img_7)
        # print (idct_7.shape)
        # idct_7_sa = dct.idct(dct_7_sa)
        img_7_1 = idct_7 * img_7_sa
        dct_7_1 = dct_img_7 * dct_7_sa
        img_7_2 = img_7 + img_7_1
        dct_7_2 = dct_7 + dct_7_1
        img_12_input = (img_7_2, 12)
        img_12 = self.backbone_img(img_12_input)
        dct_12_input = (dct_7_2, 12)
        dct_12 = self.backbone_dct(dct_12_input)
        img_12_sa = self.sa1(img_12)
        dct_12_sa = self.sa2(dct_12)
        idct_12 = DCT.idct(dct_12)
        # dct_img_12 = DCT.dct(img_12)
        img_12_1 = idct_12 * img_12_sa
        # dct_12_1 = dct_img_12 * dct_12_sa
        img_12_2 = img_12 + img_12_1
        # dct_12_2 = dct_12 + dct_12_1
        img_0_input = (img_12_2, 0)
        img_out = self.backbone_img(img_0_input)

        return img_out