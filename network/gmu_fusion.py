import theano
from blocks import initialization
from blocks.bricks import (Initializable, FeedforwardSequence, LinearMaxout,
                           Tanh, lazy, application, BatchNormalization, Linear,
                           NDimensionalSoftmax, Logistic, Softmax, Sequence, Rectifier)
from blocks.bricks.parallel import Fork
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init


class GatedBimodal(Initializable):

    u"""Gated Bimodal neural network.
    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Logistic` brick is used.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(GatedBimodal, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]
        kwargs.setdefault('children', []).extend(self.children)
        #super(GatedBimodal, self).__init__(**kwargs)

    def _allocate(self):
        self.W = shared_floatx_nans(
            (2 * self.dim, self.dim), name='input_to_gate')
        add_role(self.W, WEIGHT)
        self.parameters.append(self.W)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @application(inputs=['x_1', 'x_2'], outputs=['output', 'z'])
    def apply(self, x_1, x_2):
        x = theano.tensor.concatenate((x_1, x_2), axis=1)
        #x = torch.cat([x_1, x_2],dim = 1)
        h = self.activation.apply(x)
        z = self.gate_activation.apply(x*self.W)
        return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:], z


class GatedClassifier(Initializable):

    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size, init_ranges, **kwargs):
        super(GatedClassifier, self).__init__(**kwargs)
        (visual_init_range, textual_init_range, gbu_init_range,
         linear_range_1, linear_range_2, linear_range_3) = init_ranges
        visual_mlp = Sequence([
            BatchNormalization(input_dim=visual_dim).apply,
            Linear(visual_dim, hidden_size, use_bias=False,
                   weights_init=initialization.Uniform(width=visual_init_range)).apply,
        ], name='visual_mlp')
        textual_mlp = Sequence([
            BatchNormalization(input_dim=textual_dim).apply,
            Linear(textual_dim, hidden_size, use_bias=False,
                   weights_init=initialization.Uniform(width=textual_init_range)).apply,
        ], name='textual_mlp')

        gbu = GatedBimodal(hidden_size,
                           weights_init=initialization.Uniform(width=gbu_init_range))

        logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size, [
                                          linear_range_1, linear_range_2, linear_range_3])
        # logistic_mlp = Sequence([
        #    BatchNormalization(input_dim=hidden_size, name='bn1').apply,
        #    Linear(hidden_size, output_dim, name='linear_output', use_bias=False,
        #           weights_init=initialization.Uniform(width=linear_range_1)).apply,
        #    Logistic().apply
        #], name='logistic_mlp')

        self.children = [visual_mlp, textual_mlp, gbu, logistic_mlp]
        kwargs.setdefault('use_bias', False)
        kwargs.setdefault('children', self.children)
        #super(GatedClassifier, self).__init__(**kwargs)

    @application(inputs=['x_v', 'x_t'], outputs=['y_hat', 'z'])
    def apply(self, x_v, x_t):
        visual_mlp, textual_mlp, gbu, logistic_mlp = self.children
        visual_h = visual_mlp.apply(x_v)
        textual_h = textual_mlp.apply(x_t)
        h, z = gbu.apply(visual_h, textual_h)
        y_hat = logistic_mlp.apply(h)
        return y_hat, z


class MLPGenreClassifier(FeedforwardSequence, Initializable):

    def __init__(self, input_dim, output_dim, hidden_size, init_ranges, output_act=Logistic, **kwargs):
        linear1 = LinearMaxout(input_dim=input_dim, output_dim=hidden_size,
                               num_pieces=2, name='linear1')
        linear2 = LinearMaxout(input_dim=hidden_size, output_dim=hidden_size,
                               num_pieces=2, name='linear2')
        linear3 = Linear(input_dim=hidden_size, output_dim=output_dim)
        logistic = output_act()
        bricks = [
            BatchNormalization(input_dim=input_dim, name='bn1'),
            linear1,
            BatchNormalization(input_dim=hidden_size, name='bn2'),
            linear2,
            BatchNormalization(input_dim=hidden_size, name='bnl'),
            linear3,
            logistic]
        for init_range, b in zip(init_ranges, (linear1, linear2, linear3)):
            b.biases_init = initialization.Constant(0)
            b.weights_init = initialization.Uniform(width=init_range)

        kwargs.setdefault('use_bias', False)
        super(MLPGenreClassifier, self).__init__(
            [b.apply for b in bricks], **kwargs)


class LinearSumClassifier(Initializable):

    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size, init_ranges, **kwargs):
        (visual_range, textual_range, linear_range_1,
         linear_range_2, linear_range_3) = init_ranges
        visual_layer = FeedforwardSequence([
            BatchNormalization(input_dim=visual_dim).apply,
            LinearMaxout(input_dim=visual_dim, output_dim=hidden_size,
                         weights_init=initialization.Uniform(
                             width=visual_range),
                         use_bias=False,
                         biases_init=initialization.Constant(0),
                         num_pieces=2).apply],
            name='visual_layer')
        textual_layer = FeedforwardSequence([
            BatchNormalization(input_dim=textual_dim).apply,
            LinearMaxout(input_dim=textual_dim, output_dim=hidden_size,
                         weights_init=initialization.Uniform(
                             width=textual_range),
                         biases_init=initialization.Constant(0),
                         use_bias=False,
                         num_pieces=2).apply],
            name='textual_layer')
        logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size, [
                                          linear_range_1, linear_range_2, linear_range_3])
        # logistic_mlp = Sequence([
        #   BatchNormalization(input_dim=hidden_size, name='bn1').apply,
        #   Linear(hidden_size, output_dim, name='linear_output', use_bias=False,
        #          weights_init=initialization.Uniform(width=linear_range_1)).apply,
        #   Logistic().apply
        #], name='logistic_mlp')

        children = [visual_layer, textual_layer, logistic_mlp]
        kwargs.setdefault('use_bias', False)
        kwargs.setdefault('children', children)
        super(LinearSumClassifier, self).__init__(**kwargs)

    @application(inputs=['x_v', 'x_t'], outputs=['y_hat'])
    def apply(self, x_v, x_t):
        visual_layer, textual_layer, logistic_mlp = self.children
        h = visual_layer.apply(x_v) + textual_layer.apply(x_t)
        return logistic_mlp.apply(h)


class ConcatenateClassifier(FeedforwardSequence, Initializable):

    def __init__(self, input_dim, output_dim, hidden_size, init_ranges, **kwargs):
        linear1 = LinearMaxout(input_dim=input_dim, output_dim=hidden_size,
                               num_pieces=2, name='linear1')
        linear2 = LinearMaxout(input_dim=hidden_size, output_dim=hidden_size,
                               num_pieces=2, name='linear2')
        linear3 = Linear(input_dim=hidden_size, output_dim=output_dim)
        logistic = Logistic()
        bricks = [
            linear1,
            BatchNormalization(input_dim=hidden_size, name='bn2'),
            linear2,
            BatchNormalization(input_dim=hidden_size, name='bnl'),
            linear3,
            logistic]
        for init_range, b in zip(init_ranges, (linear1, linear2, linear3)):
            b.biases_init = initialization.Constant(0)
            b.weights_init = initialization.Uniform(width=init_range)

        kwargs.setdefault('use_bias', False)
        super(ConcatenateClassifier, self).__init__(
            [b.apply for b in bricks], **kwargs)


class MoEClassifier(Initializable):

    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size, init_ranges, **kwargs):
        (visual_range, textual_range, linear_range_1,
         linear_range_2, linear_range_3) = init_ranges
        manager_dim = visual_dim + textual_dim
        visual_mlp = MLPGenreClassifier(visual_dim, output_dim, hidden_size, [
           linear_range_1, linear_range_2, linear_range_3], name='visual_mlp')
        textual_mlp = MLPGenreClassifier(textual_dim, output_dim, hidden_size, [
           linear_range_1, linear_range_2, linear_range_3], name='textual_mlp')
        # manager_mlp = MLPGenreClassifier(manager_dim, 2, hidden_size, [
        # linear_range_1, linear_range_2, linear_range_3], output_act=Softmax,
        # name='manager_mlp')
        bn = BatchNormalization(input_dim=manager_dim, name='bn3')
        manager_mlp = Sequence([
            Linear(manager_dim, 2, name='linear_output', use_bias=False,
                   weights_init=initialization.Uniform(width=linear_range_1)).apply,
        ], name='manager_mlp')
        fork = Fork(input_dim=manager_dim, output_dims=[2] * output_dim,
                    prototype=manager_mlp, output_names=['linear_' + str(i) for i in range(output_dim)])

        children = [visual_mlp, textual_mlp, fork, bn, NDimensionalSoftmax()]
        kwargs.setdefault('use_bias', False)
        kwargs.setdefault('children', children)
        super(MoEClassifier, self).__init__(**kwargs)

    @application(inputs=['x_v', 'x_t'], outputs=['y_hat'])
    def apply(self, x_v, x_t):
        visual_mlp, textual_mlp, fork, bn, softmax = self.children
        y_v, y_t = visual_mlp.apply(x_v), textual_mlp.apply(x_t)
        managers = fork.apply(bn.apply(theano.tensor.concatenate([x_v, x_t], axis=1)))
        g = softmax.apply(theano.tensor.stack(managers), extra_ndim=1)
        y = theano.tensor.stack([y_v, y_t])
        return (g.T * y).mean(axis=0) * 1.999 + 1e-5
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

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}
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
class Block_c(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block_c, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_filters)
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
        #print(x.shape)
        #print(self.ca(x).shape)
        x = self.ca(x) * x

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
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

        self.block1=Block_c(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block_c(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block_c(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block_c(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block_c(728,1024,2,2,start_with_relu=True,grow_first=False)

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
        if 'block7' in self.fusion:
            out['block7'] = x
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        if 'block12' in self.fusion:
            out['block12'] = x

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
        x = self.last_linear(x)
        out['x'] = x
        return out

class Block_s(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block_s, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_filters)
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
        #print(x.shape)
        #print(self.ca(x).shape)
        #x = self.ca(x) * x
        x = self.sa(x) * x

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

        self.block1 = Block_s(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block_s(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block_s(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block_s(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block_s(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        #self.GC = GatedClassifier(visual_dim=32, textual_dim=32, output_dim=32, hidden_size=32, init_ranges=(2,2,2,2,2,2))

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
        self.gb = GatedBimodal(dim = 32,weights_init=initialization.Uniform(width=2))

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
        (img, dct_layer) = input

        x = self.conv1(img)
        print(x.type)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        if 'block1' in self.fusion:
            x = x + dct_layer['block1']
        x = self.block2(x)
        if 'block2' in self.fusion:
            x = x + dct_layer['block2']
        x = self.block3(x)
        if 'block3' in self.fusion:
            x = x + dct_layer['block3']
        x = self.block4(x)
        if 'block4' in self.fusion:
            x = x + dct_layer['block4']
        x = self.block5(x)
        if 'block5' in self.fusion:
            x = x + dct_layer['block5']
        x = self.block6(x)
        if 'block6' in self.fusion:
            x = x + dct_layer['block6']
        '''if 'block1' in self.fusion:
            #x = x + dct_layer['block1']
            x = torch.cat([x,dct_layer['block1']], dim=1)
            x = self.conv_block1(x)
        x = self.block2(x)
        if 'block2' in self.fusion:
            x = torch.cat([x, dct_layer['block2']], dim=1)
            x = self.conv_block2(x)
        x = self.block3(x)
        if 'block3' in self.fusion:
            x = torch.cat([x, dct_layer['block3']], dim=1)
            x = self.conv_block3(x)
        x = self.block4(x)
        if 'block4' in self.fusion:
            x = torch.cat([x, dct_layer['block4']], dim=1)
            x = self.conv_block4(x)
        x = self.block5(x)
        if 'block5' in self.fusion:
            x = torch.cat([x, dct_layer['block5']], dim=1)
            x = self.conv_block5(x)
        x = self.block6(x)
        if 'block6' in self.fusion:
            x = torch.cat([x, dct_layer['block6']])
            x = self.conv_block6(x)'''
        x = self.block7(x)
        if 'block7' in self.fusion:
            x = x + dct_layer['block7']
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        if 'block12' in self.fusion:
            print(x.type)
            print(dct_layer['block12'].type)
            output,z = self.gb.apply(x,dct_layer['block12'])
            x = output

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
        xception1 = xception_1(pretrained=True)
        model1 = xception_img(num_classes=2,fusion=fusion)
        pretrained_dict = xception1.state_dict()
        model1_dict = model1.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model1_dict}
        model1_dict.update(pretrained_dict)
        model1.load_state_dict(model1_dict)
        model2 = xception(num_classes=2, fusion=fusion)
        pretrained_dict = xception1.state_dict()
        model2_dict = model2.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model2_dict}
        model2_dict.update(pretrained_dict)
        model2.load_state_dict(model2_dict)
        self.backbone_img = model1
        self.backbone_dct = model2

        self.post_function = nn.Softmax(dim=1)
        self.fusion = fusion

    def forward(self, input):
        #print(self.backbone_img)
        #print(self.backbone_dct)
        (image, dct) = input
        dct_out = self.backbone_dct(dct)
        # print (image_out['block3'])
        # print (image_out[self.fusion].shape)
        input1 = (image, dct_out)
        img_out = self.backbone_img(input1)  # list

        return img_out
if __name__ == "__main__":
    x = torch.randn((32,3,299,299))
    y = torch.randn((32,3,299,299))
    print(x)
    print(y)
    gbu = GatedBimodal(dim = 3,weights_init=initialization.Uniform(width=2))
    z,h = gbu.apply(x,y)
    print(z)
    print(h)
    #GC = GatedClassifier(visual_dim=32, textual_dim=32, output_dim=32, hidden_size=32, init_ranges=(2,2,2,2,2,2))
    #a,b = GC.apply(x,y)