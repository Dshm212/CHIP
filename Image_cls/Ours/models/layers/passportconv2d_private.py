import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models.losses.sign_loss import SignLoss
from .hash import custom_hash
from .chameleon_hash import owner_chameleon_hash

np.random.seed(0)


class PassportPrivateBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, passport_kwargs={}):
        super().__init__()

        self.o = o

        if passport_kwargs == {}:
            print('Warning, passport_kwargs is empty')

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.key_type = passport_kwargs.get('key_type', 'random')
        self.hash = passport_kwargs.get('hash', False)
        self.chameleon = passport_kwargs.get('chameleon', False)
        self.weight = self.conv.weight

        self.alpha = passport_kwargs.get('sign_loss', 1)
        self.norm_type = passport_kwargs.get('norm_type', 'bn')

        self.init_public_bit = passport_kwargs.get('init_public_bit', True)
        self.requires_reset_key = False

        self.register_buffer('key_private', None)
        self.register_buffer('skey_private', None)

        self.init_scale(True)
        self.init_bias(True)

        norm_type = passport_kwargs.get('norm_type', 'bn')
        # self.bn0 = nn.BatchNorm2d(o, affine=False)
        # self.bn1 = nn.BatchNorm2d(o, affine=False)

        if norm_type == 'bn':
            # self.bn = nn.BatchNorm2d(o, affine=False)
            self.bn0 = nn.BatchNorm2d(o, affine=False)
            self.bn1 = nn.BatchNorm2d(o, affine=False)

        elif norm_type == 'nose_bn':
            self.bn0 = nn.BatchNorm2d(o, affine=False)
            self.bn1 = nn.BatchNorm2d(o, affine=False)

        elif norm_type == 'gn':
            self.bn0 = nn.GroupNorm(o // 16, o, affine=False)
            self.bn1 = nn.GroupNorm(o // 16, o, affine=False)

        elif norm_type == 'in':
            self.bn0 = nn.InstanceNorm2d(o, affine=False)
            self.bn1 = nn.InstanceNorm2d(o, affine=False)

        elif norm_type == 'sbn' or norm_type == 'sbn_se':
            self.bn0 = nn.BatchNorm2d(o, affine=False)

        else:
            self.bn = nn.Sequential()

        if norm_type == 'nose_bn':
            self.fc = nn.Sequential()
        elif norm_type == 'sbn':
            self.fc = nn.Sequential()
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(o, o // 4, bias=False),
                nn.LeakyReLU(inplace=True),
                # nn.Tanh(),
                nn.Linear(o // 4, o, bias=False)
            )

            self.fc2 = nn.Sequential(
                nn.Linear(o, o // 4, bias=False),
                nn.LeakyReLU(inplace=True),
                # nn.Tanh(),
                nn.Linear(o // 4, o, bias=False)
            )

        b = passport_kwargs.get('b', torch.sign(torch.rand(o) - 0.5))  # bit information to store

        if isinstance(b, int):
            b = torch.ones(o) * b
        if isinstance(b, str):
            if len(b) * 8 > o:
                raise Exception('Too much bit information')
            bsign = torch.sign(torch.rand(o) - 0.5)
            bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])

            for i, c in enumerate(bitstring):
                if c == '0':
                    bsign[i] = -1
                else:
                    bsign[i] = 1

            b = bsign

        self.register_buffer('b', b)
        self.sign_loss_private = SignLoss(self.alpha, self.b)

        self.l1_loss = nn.L1Loss()
        self.reset_parameters()

        self.reset_fc()

    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)

            # self.bias0 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            # self.bias1 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            # init.zeros_(self.bias0)
            # init.zeros_(self.bias1)

        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)

            # self.scale0 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            # self.scale1 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            # init.ones_(self.scale0)
            # init.ones_(self.scale1)

        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def reset_fc(self):
        for m in self.fc1.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
        for i in self.fc1.parameters():
            i.requires_grad = True

        for m in self.fc2.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
        for i in self.fc2.parameters():
            i.requires_grad = True

    def reset_b(self, x):
        print("Reset b")
        print(self.hash)
        print(self.chameleon)
        if self.chameleon:
            print("Using Chameleon Hash...")
            owner_signature = "Copyright to CVPR 2025"
            params, r1, hash_value, b = owner_chameleon_hash(self.skey_private, owner_signature, hash_length=self.o)

        else:
            print("Using SHA512 Hash...")
            b = custom_hash(self.skey_private, hash_length=self.o)

        if isinstance(b, int):
            b = torch.ones(self.o) * b
        if isinstance(b, str):
            if len(b) * 8 > self.o:
                raise Exception('Too much bit information')
            bsign = torch.sign(torch.rand(self.o) - 0.5)
            bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])

            for i, c in enumerate(bitstring):
                if c == '0':
                    bsign[i] = -1
                else:
                    bsign[i] = 1

            b = bsign

        b = b.to(x.device)
        self.register_buffer('b', b)
        self.sign_loss_private = SignLoss(self.alpha, self.b)

    def passport_selection(self, passport_candidates):
        b, c, h, w = passport_candidates.size()

        if c == 3:  # input channel
            randb = random.randint(0, b - 1)
            return passport_candidates[randb].unsqueeze(0)

        passport_candidates = passport_candidates.view(b * c, h, w)
        full = False
        flag = [False for _ in range(b * c)]
        channel = c
        passportcount = 0
        bcount = 0
        passport = []

        while not full:
            if bcount >= b:
                bcount = 0

            randc = bcount * channel + random.randint(0, channel - 1)
            while flag[randc]:
                randc = bcount * channel + random.randint(0, channel - 1)
            flag[randc] = True

            passport.append(passport_candidates[randc].unsqueeze(0).unsqueeze(0))

            passportcount += 1
            bcount += 1

            if passportcount >= channel:
                full = True

        passport = torch.cat(passport, dim=1)
        return passport

    def set_key(self, x, y=None):
        n = int(x.size(0))
        print("===========================")
        print(x.size())

        if n != 1:
            x = self.passport_selection(x)
            if y is not None:
                y = self.passport_selection(y)

        # assert x.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('key_private', x)

        # assert y is not None and y.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('skey_private', y)

        if self.hash and self.key_type != 'random':
            self.reset_b(x)

    def get_scale_key(self):
        return self.skey_private

    def get_scale_private(self):
        skey = self.skey_private
        scale_loss = self.sign_loss_private

        scalekey = self.conv(skey)
        b = scalekey.size(0)
        c = scalekey.size(1)

        scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        scale = scale.mean(dim=0).view(1, c, 1, 1)

        scale_for_loss = scale
        scale = scale.view(1, c)
        scale = self.fc1(scale).view(1, c, 1, 1)

        skey_size = skey.size()
        tmp = skey
        tmp = tmp.view(skey_size[0], skey_size[1], -1).mean(dim=2).view(skey_size[0], skey_size[1], 1, 1)
        tmp = tmp.mean(dim=0).view(1, -1, 1, 1)

        tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-6)
        tmp = tmp.clamp(-3, 3)

        target_channel = scale.size(1)
        current_channel = tmp.size(1)

        if current_channel >= target_channel:
            tmp = tmp[:, :target_channel]
        else:
            repeat_times = (target_channel + current_channel - 1) // current_channel
            tmp = tmp.repeat(1, repeat_times, 1, 1)
            tmp = tmp[:, :target_channel]

        scale = tmp + scale

        if scale_loss is not None:
            scale_loss.reset()
            # scale_loss.add(scale)
            scale_loss.add(scale_for_loss)

        return scale_for_loss, scale

    def get_scale_bn(self, force_passport=False, ind=0):
        if self.scale is not None and not force_passport:
            return self.scale.view(1, -1, 1, 1)
            # if ind == 0: #NOTE: this is the public branch
            #     return self.scale0.view(1, -1, 1, 1)
            # elif ind == 1:
            #     return self.scale1.view(1, -1, 1, 1)
            # else:
            #     raise ValueError

    def get_bias_key(self):
        return self.key_private

    def get_bias_private(self):
        key = self.key_private

        biaskey = self.conv(key)
        b = biaskey.size(0)
        c = biaskey.size(1)

        bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        bias = bias.mean(dim=0).view(1, c, 1, 1)

        bias = bias.view(1, c)
        bias = self.fc2(bias).view(1, c, 1, 1)

        key_size = key.shape
        tmp = key.detach().clone()
        tmp = tmp.view(key_size[0], key_size[1], -1).mean(dim=2).view(key_size[0], key_size[1], 1, 1)
        tmp = tmp.mean(dim=0).view(1, -1, 1, 1)

        tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-6)
        tmp = tmp.clamp(-3, 3)

        target_channel = bias.size(1)
        current_channel = tmp.size(1)

        if current_channel >= target_channel:
            tmp = tmp[:, :target_channel]
        else:
            repeat_times = (target_channel + current_channel - 1) // current_channel
            tmp = tmp.repeat(1, repeat_times, 1, 1)
            tmp = tmp[:, :target_channel]

        bias = tmp + bias

        return bias

    def get_bias_bn(self, force_passport=False, ind=0):
        if self.bias is not None and not force_passport:
            return self.bias.view(1, -1, 1, 1)
            # if ind == 0:
            #     return self.bias0.view(1, -1, 1, 1)
            # elif ind == 1:
            #     return self.bias1.view(1, -1, 1, 1)
            # else:
            #     raise ValueError

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keyname = prefix + 'key_private'
        skeyname = prefix + 'skey_private'

        if keyname in state_dict:
            self.register_buffer('key_private', torch.randn(*state_dict[keyname].size()))
        if skeyname in state_dict:
            self.register_buffer('skey_private', torch.randn(*state_dict[skeyname].size()))

        scalename = prefix + 'scale'
        biasname = prefix + 'bias'
        if scalename in state_dict:
            self.scale = nn.Parameter(torch.randn(*state_dict[scalename].size()))

        if biasname in state_dict:
            self.bias = nn.Parameter(torch.randn(*state_dict[biasname].size()))

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def generate_key(self, *shape):
        newshape = list(shape)
        newshape[0] = 1

        min = -1.0
        max = 1.0
        key = np.random.uniform(min, max, newshape)
        return key

    def get_loss(self):  # NOTE: this is the balance loss
        _, scale = self.get_scale_private()  # [1, c, 1, 1]
        bias = self.get_bias_private()
        scale = scale.view(-1)
        bias = bias.view(-1)
        # loss = self.l1_loss(self.scale0, self.scale1 * scale) + self.l1_loss(self.bias0, self.bias1 * scale + bias)
        loss = self.l1_loss(self.scale, scale) + self.l1_loss(self.bias, bias)
        return loss

    def forward(self, x, force_passport=False, ind=0):
        key = self.key_private
        if (key is None and self.key_type == 'random') or self.requires_reset_key:
            print("Generating key, skey, and b")
            self.set_key(torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device),
                         torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device))
            if self.hash:
                self.reset_b(x)

        scale = self.scale.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        x = self.conv(x)

        if ind == 0:
            x = self.bn0(x)
            x = scale * x + bias
        else:
            x = self.bn0(x)
            _, scale = self.get_scale_private()
            bias = self.get_bias_private()
            x = scale * x + bias

        x = self.relu(x)

        # scale0 = self.scale0.view(1, -1, 1, 1)
        # scale1 = self.scale1.view(1, -1, 1, 1)
        # bias0 = self.bias0.view(1, -1, 1, 1)
        # bias1 = self.bias1.view(1, -1, 1, 1)
        #
        # x = self.conv(x)
        #
        # if ind == 0:
        #     x = self.bn0(x)
        #     x = scale0 * x + bias0
        # else:
        #     x = self.bn1(x)
        #     x = scale1 * x + bias1
        #     _, scale = self.get_scale_private()
        #     bias = self.get_bias_private()
        #     x = scale * x + bias

        # x = self.conv(x)
        # if self.norm_type == 'bn' or self.norm_type == 'nose_bn':
        #     if ind == 0:
        #         x = self.bn0(x)
        #     else:
        #         x = self.bn1(x)
        #
        # else:
        #     x = self.bn(x)

        # scale1, scale2 = self.get_scale(force_passport, ind)
        # bias = self.get_bias(force_passport, ind)
        # x = scale2 * x + bias
        # x = self.relu(x)

        # logdir = '/data-x/g12/zhangjie/DeepIPR/ours/alexnet_cifar100_v2_all-random-our/passport_attack_3_random'
        # log_file = os.path.join(logdir, 'scale_random.txt')
        # lf = open(log_file, 'a')
        #
        # print("SCALE1", scale1, file =lf)
        # print("##############################################", file=lf)
        # print("SCALE2", scale2, file=lf)
        # print("##############################################",file=lf)
        # lf.flush()

        return x
