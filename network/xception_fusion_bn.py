"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, fusion = 'fc'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.fusion = fusion

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
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

    def features(self, input):
        out = {}
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        if 'block1' in self.fusion:
            out['block1'] = x
        x = self.block2(x)
        if 'block2' in self.fusion:
            out['block2'] = x
        x = self.block3(x)
        if 'block3' in self.fusion:
            out['block3'] = x
        x = self.block4(x)
        if 'block4' in self.fusion:
            out['block4'] = x
        x = self.block5(x)
        if 'block5' in self.fusion:
            out['block5'] = x
        x = self.block6(x)
        if 'block6' in self.fusion:
            out['block6'] = x
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
        out['x'] = self.bn4(x)
        return out

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        out = {}
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        if 'block1' in self.fusion:
            out['block1'] = x
        x = self.block2(x)
        if 'block2' in self.fusion:
            out['block2'] = x
        x = self.block3(x)
        if 'block3' in self.fusion:
            out['block3'] = x
        x = self.block4(x)
        if 'block4' in self.fusion:
            out['block4'] = x
        x = self.block5(x)
        if 'block5' in self.fusion:
            out['block5'] = x
        x = self.block6(x)
        if 'block6' in self.fusion:
            out['block6'] = x
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
        if 'fc' in self.fusion:
            out['fc'] = x
        out['x'] = self.last_linear(x)
        return out


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
        self.conv_block1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv_block2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.conv_block3 = nn.Conv2d(1456, 728, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.conv_block4 = nn.Conv2d(1456, 728, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.conv_block5 = nn.Conv2d(1456, 728, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.conv_block6 = nn.Conv2d(1456, 728, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.conv_bn1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv_bn2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv_bn3 = nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv_bn4 = nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv_bn5 = nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv_bn6 = nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.block1_bn = nn.BatchNorm2d(128)
        self.block2_bn = nn.BatchNorm2d(256)
        self.block3_bn = nn.BatchNorm2d(728)
        self.block4_bn = nn.BatchNorm2d(728)
        self.block5_bn = nn.BatchNorm2d(728)
        self.block6_bn = nn.BatchNorm2d(728)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        (img, dct_layer) = input

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        if 'block1' in self.fusion:
            #print (dct_layer['block1'].shape)
            dct = self.conv_bn1(dct_layer['block1'])
            #print (dct.shape)
            dct = self.block1_bn(dct)
            #print (dct.shape)
            dct = self.relu(dct)
            #print (x.shape)
            #print (dct.shape)
            x = torch.cat([x,dct], dim=1)
            x = self.conv_block1(x)
        x = self.block2(x)
        if 'block2' in self.fusion:
            dct = self.conv_bn2(dct_layer['block2'])
            dct = self.block2_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block2(x)
        x = self.block3(x)
        if 'block3' in self.fusion:
            dct = self.conv_bn3(dct_layer['block3'])
            dct = self.block3_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block3(x)
        x = self.block4(x)
        if 'block4' in self.fusion:
            dct = self.conv_bn4(dct_layer['block4'])
            dct = self.block4_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block4(x)
        x = self.block5(x)
        if 'block5' in self.fusion:
            dct = self.conv_bn5(dct_layer['block5'])
            dct = self.block5_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block5(x)
        x = self.block6(x)
        if 'block6' in self.fusion:
            dct = self.conv_bn6(dct_layer['block6'])
            dct = self.block6_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block6(x)
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
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        
        (img, dct_layer) = input

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        if 'block1' in self.fusion:
            #print (dct_layer['block1'].shape)
            dct = self.conv_bn1(dct_layer['block1'])
            #print (dct.shape)
            dct = self.block1_bn(dct)
            #print (dct.shape)
            dct = self.relu(dct)
            #print (x.shape)
            #print (dct.shape)
            x = torch.cat([x,dct], dim=1)
            x = self.conv_block1(x)
        x = self.block2(x)
        if 'block2' in self.fusion:
            dct = self.conv_bn2(dct_layer['block2'])
            dct = self.block2_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block2(x)
        x = self.block3(x)
        if 'block3' in self.fusion:
            dct = self.conv_bn3(dct_layer['block3'])
            dct = self.block3_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block3(x)
        x = self.block4(x)
        if 'block4' in self.fusion:
            dct = self.conv_bn4(dct_layer['block4'])
            dct = self.block4_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block4(x)
        x = self.block5(x)
        if 'block5' in self.fusion:
            dct = self.conv_bn5(dct_layer['block5'])
            dct = self.block5_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block5(x)
        x = self.block6(x)
        if 'block6' in self.fusion:
            dct = self.conv_bn6(dct_layer['block6'])
            dct = self.block6_bn(dct)
            dct = self.relu(dct)
            x = torch.cat([x, dct], dim=1)
            x = self.conv_block6(x)
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
        if 'fc' in self.fusion:
            x = x + dct_layer['fc']
            #print('fc')
        x = self.last_linear(x)
        return x

def xception(num_classes=1000, pretrained=False, fusion='fc'):
    model = Xception(num_classes=num_classes,fusion=fusion)
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

class Img_DCT_BlockFusion(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, fusion=('fc')):
        super(Img_DCT_BlockFusion, self).__init__()
        self.backbone_img = xception_img(num_classes=num_classes, pretrained=pretrained, fusion=fusion)
        self.backbone_dct = xception(num_classes=num_classes, pretrained=pretrained, fusion=fusion)

        self.post_function = nn.Softmax(dim=1)
        self.fusion = fusion

    def forward(self, input):
        (image, dct) = input
        dct_out = self.backbone_dct(dct)
        # print (image_out['block3'])
        # print (image_out[self.fusion].shape)
        input1 = (image, dct_out)
        img_out = self.backbone_img(input1)  # list

        return img_out