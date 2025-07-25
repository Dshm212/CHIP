import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d import PassportBlock


def get_convblock(passport_kwargs):
    def convblock_(*args, **kwargs):
        if passport_kwargs['flag']:
            return PassportBlock(*args, **kwargs, passport_kwargs=passport_kwargs)
        else:
            return ConvBlock(*args, **kwargs, bn=passport_kwargs['norm_type'])

    return convblock_


class BasicPassportBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, passport_kwargs={}):
        super(BasicPassportBlock, self).__init__()

        self.convbnrelu_1 = get_convblock(passport_kwargs['convbnrelu_1'])(in_planes, planes, 3, stride, 1)
        self.convbn_2 = get_convblock(passport_kwargs['convbn_2'])(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_convblock(passport_kwargs['shortcut'])(in_planes, self.expansion * planes, 1, stride, 0)

    def set_intermediate_keys(self, pretrained_block, x, y=None):
        if isinstance(self.convbnrelu_1, PassportBlock):
            self.convbnrelu_1.set_key(x, y)
        out_x = pretrained_block.convbnrelu_1(x)
        if y is not None:
            out_y = pretrained_block.convbnrelu_1(y)
        else:
            out_y = None

        if isinstance(self.convbn_2, PassportBlock):
            self.convbn_2.set_key(out_x, out_y)
        out_x = pretrained_block.convbn_2(out_x)
        if y is not None:
            out_y = pretrained_block.convbn_2(out_y)

        if not isinstance(self.shortcut, nn.Sequential):
            if isinstance(self.shortcut, PassportBlock):
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

    def forward(self, x, force_passport=False):
        if isinstance(self.convbnrelu_1, PassportBlock):
            out = self.convbnrelu_1(x, force_passport)
        else:
            out = self.convbnrelu_1(x)
        if isinstance(self.convbnrelu_1, PassportBlock):
            out = self.convbn_2(out, force_passport)
        else:
            out = self.convbn_2(out)

        if not isinstance(self.shortcut, nn.Sequential):
            if isinstance(self.shortcut, PassportBlock):
                out = out + self.shortcut(x, force_passport)
            else:
                out = out + self.shortcut(x)
        else:
            out = out + x
        out = F.relu(out)
        return out


class ResNetPassport(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, passport_kwargs={}):
        super(ResNetPassport, self).__init__()
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
            if isinstance(self.convbnrelu_1, PassportBlock):
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

    def forward(self, x, force_passport=False):
        if isinstance(self.convbnrelu_1, PassportBlock):
            out = self.convbnrelu_1(x, force_passport)
        else:
            out = self.convbnrelu_1(x)
        for block in self.layer1:
            out = block(out, force_passport)
        for block in self.layer2:
            out = block(out, force_passport)
        for block in self.layer3:
            out = block(out, force_passport)
        for block in self.layer4:
            out = block(out, force_passport)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet18Passport(**model_kwargs):
    return ResNetPassport(BasicPassportBlock, [2, 2, 2, 2], **model_kwargs)


if __name__ == '__main__':
    import json
    from pprint import pprint

    passport_settings = json.load(open('../passport_configs/resnet_passport_l2.json'))
    passport_kwargs = {}

    for layer_key in passport_settings:
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:
                passport_kwargs[layer_key][i] = {}
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]:
                    flag = passport_settings[layer_key][i][module_key]
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True
                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag,
                        'norm_type': 'gn',
                        'key_type': 'random',
                        'sign_loss': 1
                    }
                    if b is not None:
                        passport_kwargs[layer_key][i][module_key]['b'] = b

        else:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            passport_kwargs[layer_key] = {
                'flag': flag,
                'norm_type': 'gn',
                'key_type': 'random',
                'sign_loss': 1
            }
            if b is not None:
                passport_kwargs[layer_key][i][module_key]['b'] = b

    pprint(passport_kwargs)
    key_model = ResNet18Passport(passport_kwargs=passport_kwargs)
    for name in key_model.named_modules():
        print(name[0], name[1].__class__.__name__)

    key_model.set_intermediate_keys(ResNet18Passport(passport_kwargs=passport_kwargs),
                                    torch.randn(1, 3, 32, 32),
                                    torch.randn(1, 3, 32, 32))

    key_model(torch.randn(1, 3, 32, 32))