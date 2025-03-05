import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d_private_ERB import PassportPrivateBlockERB


def get_convblock(passport_kwargs):
    def convblock_(*args, **kwargs):
        if passport_kwargs['flag']:
            return PassportPrivateBlockERB(*args, **kwargs, passport_kwargs=passport_kwargs)
        else:
            return ConvBlock(*args, **kwargs, bn=passport_kwargs['norm_type'])

    return convblock_


class BasicPrivateBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, passport_kwargs={}):
        super(BasicPrivateBlock, self).__init__()
        # self.expansion = 1

        self.convbnrelu_1 = get_convblock(passport_kwargs['convbnrelu_1'])(in_planes, planes, 3, stride, 1)
        self.convbn_2 = get_convblock(passport_kwargs['convbn_2'])(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_convblock(passport_kwargs['shortcut'])(in_planes, self.expansion * planes, 1, stride, 0)

    def set_intermediate_keys(self, pretrained_block, x, y=None):

        if isinstance(self.convbnrelu_1, PassportPrivateBlockERB):
            self.convbnrelu_1.set_key(x, y)

        out_x = pretrained_block.convbnrelu_1(x)
        if y is not None:
            out_y = pretrained_block.convbnrelu_1(y)
        else:
            out_y = None

        if isinstance(self.convbn_2, PassportPrivateBlockERB):
            self.convbn_2.set_key(out_x, out_y)
        out_x = pretrained_block.convbn_2(out_x)

        if y is not None:
            out_y = pretrained_block.convbn_2(out_y)

        if not isinstance(self.shortcut, nn.Sequential):
            if isinstance(self.shortcut, PassportPrivateBlockERB):
                self.shortcut.set_key(x, y)

            shortcut_x = pretrained_block.shortcut(x)
            out_x = out_x + shortcut_x
            if y is not None:
                shortcut_y = pretrained_block.shortcut(y)
                out_y = out_y + shortcut_y
        else:
            out_x = out_x + x
            if y is not None:
                out_y = out_y + y

        out_x = F.relu(out_x)
        if y is not None:
            out_y = F.relu(out_y)

        return out_x, out_y

    def forward(self, x, force_passport=False, ind=0):
        if isinstance(self.convbnrelu_1, PassportPrivateBlockERB):
            out = self.convbnrelu_1(x, force_passport, ind)
        else:
            out = self.convbnrelu_1(x)

        if isinstance(self.convbn_2, PassportPrivateBlockERB):
            out = self.convbn_2(out, force_passport, ind)
        else:
            out = self.convbn_2(out)

        if not isinstance(self.shortcut, nn.Sequential):
            if isinstance(self.shortcut, PassportPrivateBlockERB):
                out = out + self.shortcut(x, force_passport, ind)
            else:
                out = out + self.shortcut(x)
        else:
            out = out + x
        out = F.relu(out)
        return out


class ResNetPrivate(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, passport_kwargs={}):
        super(ResNetPrivate, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks

        self.convbnrelu_1 = get_convblock(passport_kwargs['convbnrelu_1'])(3, 64, 3, 1, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, passport_kwargs=passport_kwargs['layer1'])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, passport_kwargs=passport_kwargs['layer2'])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, passport_kwargs=passport_kwargs['layer3'])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, passport_kwargs=passport_kwargs['layer4'])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, passport_kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, passport_kwargs[str(i)]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_intermediate_keys(self, pretrained_model, x, y=None):
        with torch.no_grad():
            if isinstance(self.convbnrelu_1, PassportPrivateBlockERB):
                self.convbnrelu_1.set_key(x, y)

            x = pretrained_model.convbnrelu_1(x)
            if y is not None:
                y = pretrained_model.convbnrelu_1(y)

            for self_block, pretrained_block in zip(self.layer1, pretrained_model.layer1):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)
            for self_block, pretrained_block in zip(self.layer2, pretrained_model.layer2):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)
            for self_block, pretrained_block in zip(self.layer3, pretrained_model.layer3):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)
            for self_block, pretrained_block in zip(self.layer4, pretrained_model.layer4):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)

    def forward(self, x, force_passport=False, ind=0):
        if isinstance(self.convbnrelu_1, PassportPrivateBlockERB):
            out = self.convbnrelu_1(x, force_passport, ind)
        else:
            out = self.convbnrelu_1(x)

        for block in self.layer1:
            out = block(out, force_passport, ind)
        for block in self.layer2:
            out = block(out, force_passport, ind)
        for block in self.layer3:
            out = block(out, force_passport, ind)
        for block in self.layer4:
            out = block(out, force_passport, ind)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet18Private(**model_kwargs):
    return ResNetPrivate(BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs)