import copy
import json
import os
import sys
import torch
import torch.optim as optim
import collections
import passport_generator
import wandb
from experiments.base import Experiment
from experiments.trainer import Trainer
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from experiments.utils import construct_passport_kwargs

from dataset import prepare_dataset, prepare_wm, prepare_ERB_dataset, prepare_attack_dataset

from models.layers.conv2d import ConvBlock
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.resnet_normal import ResNet18
from models.resnet_passport_private import ResNet18Private

# for prun
import shutil
import matplotlib.pyplot as plt
from experiments.logger import Logger, savefig
from prun import test, test_signature, pruning_resnet, pruning_resnet2
import numpy as np

# for ambiguity
from amb_attack import train_maximize, test_fake, train_ERB
import pandas as pd
import torch.nn as nn
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.layers.passportconv2d_private_ERB import PassportPrivateBlockERB
import torch.nn.functional as F

from models.layers.hash import custom_hash
from torch.optim.lr_scheduler import LambdaLR


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)

class ClassificationPrivateExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.in_channels = 1 if self.dataset == 'mnist' else 3
        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256
        }[self.dataset]

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

        self.train_data, self.valid_data = prepare_dataset(self.args)
        self.wm_data = None

        if self.use_trigger_as_passport:
            self.passport_data = prepare_wm('data/trigger_set/pics')
        else:
            self.passport_data = self.valid_data

        if self.train_backdoor:
            self.wm_data = prepare_wm('data/trigger_set/pics')

        self.construct_model()

        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        self.trainer = TrainerPrivate(self.model, optimizer, scheduler, self.device, self.dataset, self.scheme)

        if self.is_tl:
            self.finetune_load()
        else:
            self.makedirs_or_load()

    def construct_model(self):
        def setup_keys():
            # if self.key_type != 'random':
            if self.arch == 'alexnet':
                pretrained_model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
            else:
                pretrained_model = ResNet18(num_classes=self.num_classes, norm_type=self.norm_type)
            print('Loading pretrained model from', self.pretrained_path)
            pretrained_dict = torch.load(self.pretrained_path)
            filtered_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k and 'linear' not in k}
            pretrained_model.load_state_dict(filtered_dict, strict=False)
            # pretrained_model.load_state_dict(torch.load(self.pretrained_path))
            pretrained_model = pretrained_model.to(self.device)
            self.setup_keys(pretrained_model)

        passport_kwargs = construct_passport_kwargs(self)
        self.passport_kwargs = passport_kwargs

        if self.arch == 'alexnet':
            model = AlexNetPassportPrivate(self.in_channels, self.num_classes, passport_kwargs)
        else:
            model = ResNet18Private(num_classes=self.num_classes, passport_kwargs=passport_kwargs)

        self.model = model.to(self.device)
        # self.copy_model = model.to(self.device)

        setup_keys()

    def setup_keys(self, pretrained_model):
        if self.key_type != 'random':
            n = 1 if self.key_type == 'image' else 20  # any number

            key_x, x_inds = passport_generator.get_key(self.passport_data, n)
            # key_x, x_inds = passport_generator.get_key_sig(self.passport_data, n)
            key_x = key_x.to(self.device)
            key_y, y_inds = passport_generator.get_key(self.passport_data, n)
            # key_y, y_inds = passport_generator.get_key_sig(self.passport_data, n)
            key_y = key_y.to(self.device)

            passport_generator.set_key(pretrained_model, self.model,
                                       key_x, key_y)
        else:
            batch = next(iter(self.passport_data))
            inputs = batch[0].to(self.device)
            # forward for once to set random key and s_key, and then generate the hashed b (signature)
            outputs = self.model(inputs)

    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        best_file = os.path.join(self.logdir, 'best.txt')
        last_file = os.path.join(self.logdir, 'last.json')
        best_ep = 1

        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        task_name = f'{str(4090)}_{self.dataset}_{self.arch}_v{self.scheme}_{self.tag}_{self.norm_type}'

        run = wandb.init(
            entity="dshm",
            project="CHIP",
            name=task_name,
            config={
                "arch": self.arch,
                "dataset": self.dataset,
                "tl dataset": self.tl_dataset,
                "epochs": self.epochs,
                "batch size": self.batch_size,
                "learning rate": self.lr,
                "tag": self.tag,
                "norm type": self.norm_type,
                "train passport": self.train_passport,
                "train private": self.train_private,
                "train backdoor": self.train_backdoor,
                "hash": self.hash,
                "chameleon": self.chameleon,
                "passport config": self.passport_config
            },
        )

        for ep in range(1, self.epochs + 1):
            train_metrics = self.trainer.train(ep, self.train_data, self.wm_data, self.arch)
            print(f'Sign Detection Accuracy: {train_metrics["sign_acc"] * 100:6.4f}')

            valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

            wm_metrics = {}
            if self.train_backdoor:
                wm_metrics = self.trainer.test(self.wm_data, 'WM Result')
                # f.write(str(wm_metrics) +'\n')

            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'wm_{key}'] = wm_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_total_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_total_acc']
                self.save_model('best.pth')
                best_ep = ep

            self.save_last_model()

            f = open(best_file, 'a')
            f.write(str(best_acc) + "\n")
            print(str(wm_metrics) + '\n', file=f)
            print(str(metrics) + '\n', file=f)
            f.write("\n")
            f.write("best epoch: %s" % str(best_ep) + '\n')
            f.flush()

            with open(last_file, 'w', encoding='utf-8') as f:
                json.dump(valid_metrics, f, indent=4)

            run.log({**metrics, "epoch": ep})

        run.finish()

    def evaluate(self):
        self.trainer.test(self.valid_data)

    def transfer_learning(self):
        if not self.is_tl:
            raise Exception('Please run with --transfer-learning')

        if self.tl_dataset == 'caltech-101':
            self.num_classes = 101
        elif self.tl_dataset == 'cifar100':
            self.num_classes = 100
        elif self.tl_dataset == 'caltech-256':
            self.num_classes = 257
        else:  # cifar10
            self.num_classes = 10

        # load clone model
        print('Loading clone model')
        # if self.arch == 'alexnet':
        #     tl_model = AlexNetNormal(self.in_channels,
        #                              self.num_classes,
        #                              self.norm_type)
        # else:
        #     tl_model = ResNet18(num_classes=self.num_classes,
        #                         norm_type=self.norm_type)

        # ??????fine-tune alex ??
        passport_kwargs = construct_passport_kwargs(self)

        if self.arch == 'alexnet':
            tl_model = AlexNetPassportPrivate(self.in_channels, self.num_classes, passport_kwargs)
        else:
            tl_model = ResNet18Private(num_classes=self.num_classes, passport_kwargs=passport_kwargs)

        ##### load / reset weights of passport layers for clone model #####

        try:
            tl_model.load_state_dict(self.model.state_dict())
            # tl_model.load_state_dict(self.copy_model.state_dict())
        except:
            print('Having problem to direct load state dict, loading it manually')
            if self.arch == 'alexnet':
                for tl_m, self_m in zip(tl_model.features, self.model.features):

                    try:
                        tl_m.load_state_dict(self_m.state_dict())
                    except:
                        print(
                            'Having problem to load state dict usually caused by missing keys, load by strict=False')
                        tl_m.load_state_dict(self_m.state_dict(), False)  # load conv weight, bn running mean
                        # print(self_m)
                        # print(tl_m)
                        # ???????
                        # tl_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                        # tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

                        # ?????bn??
                        scale1, scale2 = self_m.get_scale()
                        tl_m.bn.weight.data.copy_(scale1.detach().view(-1))
                        tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))



            else:

                passport_settings = self.passport_config
                for l_key in passport_settings:  # layer
                    if isinstance(passport_settings[l_key], dict):
                        for i in passport_settings[l_key]:  # sequential
                            for m_key in passport_settings[l_key][i]:  # convblock

                                tl_m = tl_model.__getattr__(l_key)[int(i)].__getattr__(m_key)  # type: ConvBlock
                                self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)

                                try:
                                    tl_m.load_state_dict(self_m.state_dict())
                                except:
                                    print(f'{l_key}.{i}.{m_key} cannot load state dict directly')
                                    # print(self_m)
                                    # print(tl_m)
                                    tl_m.load_state_dict(self_m.state_dict(), False)

                                    scale1, scale2 = self_m.get_scale()
                                    tl_m.bn.weight.data.copy_(scale1.detach().view(-1))
                                    tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

                    else:
                        print("FFFFFFFFFFFFFFFFFFFFFFF")
                        tl_m = tl_model.__getattr__(l_key)
                        self_m = self.model.__getattr__(l_key)

                        try:
                            tl_m.load_state_dict(self_m.state_dict())
                        except:
                            print(f'{l_key} cannot load state dict directly')
                            tl_m.load_state_dict(self_m.state_dict(), False)
                            # tl_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))

                            scale1, scale2 = self_m.get_scale()
                            tl_m.bn.weight.data.copy_(scale1.detach().view(-1))
                            tl_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

        tl_model.to(self.device)
        print('Loaded clone model')

        # tl scheme setup
        # if self.tl_scheme == 'rtal':
        #     # rtal = reset last layer + train all layer
        #     # ftal = train all layer
        #     try:
        #         tl_model.classifier.reset_parameters()
        #     except:
        #         tl_model.linear.reset_parameters()

        if self.tl_scheme == 'rtal':
            # rtal = reset last layer + train all layer
            # ftal = train all layer
            try:
                if isinstance(tl_model.classifier, nn.Sequential):
                    tl_model.classifier[-1].reset_parameters()
                else:
                    tl_model.classifier.reset_parameters()
            except:
                tl_model.linear.reset_parameters()

        # for name, m in tl_model.named_modules():
        #     if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
        #         print(name)
        #         print(m.key_private.requires_grad, m.skey_private.requires_grad, m.b.requires_grad)
        #         for param in m.fc.parameters():
        #             print(param.requires_grad)
        #             param.requires_grad = False
        #             print(param.requires_grad)

        optimizer = optim.SGD(tl_model.parameters(),
                              # lr=self.lr,
                              lr=0.001,
                              momentum=0.9,
                              weight_decay=0.0005)

        # optimizer = optim.Adam(tl_model.parameters(),
        #                         lr=0.001,
        #                         weight_decay=0.0005
        #                         )

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        tl_trainer = Trainer(tl_model,
                             optimizer,
                             scheduler,
                             self.device)

        tester = TesterPrivate(tl_model,
                               self.device)

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True
        best_acc = 0
        best_file = os.path.join(self.logdir, 'best.txt')
        best_ep = 1

        for ep in range(1, self.epochs + 1):
            train_metrics = tl_trainer.train(ep, self.train_data)
            valid_metrics = tl_trainer.test(self.valid_data)

            # ##### load transfer learning weights from clone model  #####
            # try:
            #     self.model.load_state_dict(tl_model.state_dict())
            # except:
            #     if self.arch == 'alexnet':
            #         for tl_m, self_m in zip(tl_model.features, self.model.features):
            #             try:
            #                 self_m.load_state_dict(tl_m.state_dict())
            #             except:
            #                 self_m.load_state_dict(tl_m.state_dict(), False)
            #     else:
            #         passport_settings = self.passport_config
            #         for l_key in passport_settings:  # layer
            #             if isinstance(passport_settings[l_key], dict):
            #                 for i in passport_settings[l_key]:  # sequential
            #                     for m_key in passport_settings[l_key][i]:  # convblock
            #                         tl_m = tl_model.__getattr__(l_key)[int(i)].__getattr__(m_key)
            #                         self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)
            #
            #                         try:
            #                             self_m.load_state_dict(tl_m.state_dict())
            #                         except:
            #                             self_m.load_state_dict(tl_m.state_dict(), False)
            #             else:
            #                 tl_m = tl_model.__getattr__(l_key)
            #                 self_m = self.model.__getattr__(l_key)
            #
            #                 try:
            #                     self_m.load_state_dict(tl_m.state_dict())
            #                 except:
            #                     self_m.load_state_dict(tl_m.state_dict(), False)

            wm_metrics = tester.test_signature()
            print("==============================")
            print(wm_metrics)

            L = len(wm_metrics)
            S = sum(wm_metrics.values())
            pri_sign = S / L

            if self.train_backdoor:
                backdoor_metrics = tester.test(self.wm_data, 'Old WM Accuracy')

            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'old_wm_{key}'] = wm_metrics[key]
            if self.train_backdoor:
                for key in backdoor_metrics: metrics[f'backdoor_{key}'] = backdoor_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')
                self.save_model(f'tl-epoch-{ep}.pth', tl_model)

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')
                self.save_model('tl-best.pth', tl_model)
                best_ep = ep

            self.save_last_model()
            f = open(best_file, 'a')
            print(str(wm_metrics) + '\n', file=f)
            print(str(metrics) + '\n', file=f)
            f.write('Bset ACC %s' % str(best_acc) + "\n")
            print('Private Sign Detction:', str(pri_sign) + '\n', file=f)
            f.write("\n")
            f.write("best epoch: %s" % str(best_ep) + '\n')
            f.flush()

    def pruning(self):
        device = self.device
        logdir = self.logdir
        load_path = logdir
        # print('self.logdir', self.logdir)
        prun_dir = self.logdir + '/prun'
        if not os.path.exists(prun_dir):
            os.mkdir(prun_dir)

        title = ''
        txt_pth = os.path.join(prun_dir, 'log_prun.txt')
        logger_prun = Logger(txt_pth, title=title)
        # logger_prun.set_names(['Deployment', 'Verification', 'Signature', 'Diff'])
        logger_prun.set_names(['Deployment, Norm', 'Verification, Norm', 'Signature, Norm',
                               'Deployment, Random', 'Verification, Random', 'Signature, Random'])

        txt_pth2 = os.path.join(prun_dir, 'log_prun2.txt')
        logger_prun2 = Logger(txt_pth2, title=title)
        logger_prun2.set_names(['Deployment', 'Verification', 'Signature'])

        self.train_data, self.valid_data = prepare_dataset(self.args)
        print('loadpath--------', load_path)

        for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            sd = torch.load(load_path + '/models/best.pth')
            model_copy = copy.deepcopy(self.model)
            model_copy2 = copy.deepcopy(self.model)

            model_copy.load_state_dict(sd)
            model_copy2.load_state_dict(sd)
            model_copy.to(self.device)
            model_copy2.to(self.device)
            pruning_resnet2(model_copy, perc, type_prune='l1')
            pruning_resnet2(model_copy2, perc, type_prune='rd')

            res = {}
            res2 = {}
            res_wm = {}

            # self.wm_data = prepare_wm('data/trigger_set/pics')
            res['perc'] = perc
            res['pub_ori'] = test(model_copy, device, self.valid_data, msg='pruning %s percent Dep. Result' % perc,
                                  ind=0)
            res['pri_ori'] = test(model_copy, device, self.valid_data, msg='pruning %s percent Ver. Result' % perc,
                                  ind=1)
            _, res['pri_sign_acc'] = test_signature(model_copy)

            res2['perc'] = perc
            res2['pub_ori'] = test(model_copy2, device, self.valid_data, msg='pruning %s percent Dep. Result' % perc,
                                   ind=0)
            res2['pri_ori'] = test(model_copy2, device, self.valid_data, msg='pruning %s percent Ver. Result' % perc,
                                   ind=1)
            _, res2['pri_sign_acc'] = test_signature(model_copy2)
            # res_wm['pri_ori'] = test(self.model, device, self.wm_data, msg='pruning %s percent Pri_Trigger Result' % perc, ind=1)
            del model_copy, model_copy2

            pub_acc = res['pub_ori']['acc']
            pri_acc = res['pri_ori']['acc']
            # pri_acc_wm = res_wm['pri_ori']['acc']
            pri_sign_acc = res['pri_sign_acc'] * 100

            pub_acc2 = res2['pub_ori']['acc']
            pri_acc2 = res2['pri_ori']['acc']
            pri_sign_acc2 = res2['pri_sign_acc'] * 100

            # diff = torch.abs(pub_acc - pri_acc)
            logger_prun.append([pub_acc, pri_acc, pri_sign_acc, pub_acc2, pri_acc2, pri_sign_acc2])
            # logger_prun.append([pub_acc, pri_acc, pri_sign_acc])
            # logger_prun2.append([pub_acc2, pri_acc2, pri_sign_acc2])

    def fake_attack(self):
        epochs = 5
        lr = 1e-2
        device = self.device

        loadpath = self.logdir
        print("==========================")
        print(loadpath)
        print(self.type)

        # if "ERB" in self.type:
        #     from models.resnet_passport_private_ERB import ResNet18Private
        #     from models.alexnet_passport_private_ERB import AlexNetPassportPrivate
        #
        #     if self.arch == 'alexnet':
        #         model = AlexNetPassportPrivate(self.in_channels, self.num_classes, self.passport_kwargs)
        #     else:
        #         model = ResNet18Private(num_classes=self.num_classes, passport_kwargs=self.passport_kwargs)
        #
        #     self.model = model

        task_name = loadpath.split('/')[-2]
        loadpath_all = loadpath + '/models/last.pth'
        sd = torch.load(loadpath_all)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)

        # for name, m in self.model.named_modules():
        #     if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
        #         print("========================>>>>>>")
        #         print(name)
        #
        #         keyname = 'key_private'
        #         skeyname = 'skey_private'
        #         bname = 'b'
        #
        #         key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()
        #         old_b = m.__getattr__(bname).data.clone()
        #
        #         o = old_b.size(0)
        #         new_b = custom_hash(skey, o).to(device)
        #         # print(key.size(), skey.size(), old_b.size(), new_b.size())
        #
        #         error_rate = torch.sum(torch.abs(new_b - old_b)) / old_b.size(0)
        #         print(error_rate)

        basedir = './passport_attack/' + task_name + '/'
        logdir = basedir + self.type + '_' + str(self.idx + 1)
        os.makedirs(logdir, exist_ok=True)
        best_file = os.path.join(logdir, 'best.txt')
        log_file = os.path.join(logdir, 'log.txt')
        lf = open(log_file, 'a')
        shutil.copy('amb_attack.py', str(logdir) + "/amb_attack.py")

        for param in self.model.parameters():
            param.requires_grad_(False)

        passblocks = []
        origpassport = []
        fakepassport = []
        fcparams = []

        for m in self.model.modules():
            # print(isinstance(m, PassportPrivateBlock))
            if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                # if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlockERB):

                passblocks.append(m)

                keyname = 'key_private'
                skeyname = 'skey_private'

                _, old_scale = m.get_scale_private()
                old_bias = m.get_bias_private()
                # old_scale = m.get_scale_bn()
                # old_bias = m.get_bias_bn()

                m.register_parameter('old_scale', nn.Parameter(old_scale))
                m.register_parameter('old_bias', nn.Parameter(old_bias))

                m.old_scale.requires_grad = False
                m.old_bias.requires_grad = False

                key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()

                print("============================")
                print(key.size(), skey.size())
                origpassport.append(key.to(device))
                origpassport.append(skey.to(device))

                m.__delattr__(keyname)
                m.__delattr__(skeyname)

                print("Using the original key and skey for ambiguity attack")

                # noise1 = torch.randn_like(key)
                # noise2 = torch.randn_like(skey)
                #
                # new_key = key + noise1 * 0.01
                # new_skey = skey + noise2 * 0.01
                #
                # m.register_parameter(keyname, nn.Parameter(new_key))
                # m.register_parameter(skeyname, nn.Parameter(new_skey))

                # key_mean = key.mean()
                # key_std = key.std()
                # skey_mean = skey.mean()
                # skey_std = skey.std()
                #
                # print("key stats")
                # print(key_mean, key_std)
                #
                # print("skey stats")
                # print(skey_mean, skey_std)

                new_key = torch.normal(mean=0, std=3, size=key.size())
                new_skey = torch.normal(mean=0, std=3, size=skey.size())

                m.register_parameter(keyname, nn.Parameter(new_key))
                m.register_parameter(skeyname, nn.Parameter(new_skey))

                fakepassport.append(m.__getattr__(keyname))
                fakepassport.append(m.__getattr__(skeyname))

                for layer in m.fc.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        for p in layer.parameters():
                            p.requires_grad = True

                # if "ERB" in self.type:
                #     print("ERB FC layers are trainable!")
                #     for layer in m.ERB_fc.modules():
                #         print(layer)
                #         if isinstance(layer, nn.Linear):
                #             nn.init.xavier_normal_(layer.weight)
                #             nn.init.constant_(layer.bias, 0)
                #         for p in layer.parameters():
                #             p.requires_grad = True

        if self.flipperc != 0:
            print(f'Reverse {self.flipperc * 100:.2f}% of binary signature')
            for m in passblocks:
                mflip = self.flipperc

                oldb = m.sign_loss_private.b
                newb = oldb.clone()

                npidx = np.arange(len(oldb))
                randsize = int(oldb.view(-1).size(0) * mflip)
                randomidx = np.random.choice(npidx, randsize, replace=False)

                newb[randomidx] = oldb[randomidx] * -1

                m.sign_loss_private.set_b(newb)
                m.b = newb

        # for m in passblocks:
        #     oldb = m.sign_loss_private.b
        #     newb = custom_hash(m.skey_private, oldb.size(0)).to(device)
        #     m.sign_loss_private.set_b(newb)
        #     m.b = newb

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        self.model.to(device)

        print("*************************************")

        for p_name, p in self.model.named_parameters():
            if p.requires_grad:
                print(p_name)

        criterion = nn.CrossEntropyLoss()

        history = []

        def run_cs():
            cs = []

            for d1, d2 in zip(origpassport, fakepassport):
                d1 = d1.view(d1.size(0), -1)
                d2 = d2.view(d2.size(0), -1)

                cs.append(F.cosine_similarity(d1, d2).item())

            return cs

        print('Before training', file=lf)

        res = {}
        valres = test_fake(self.model, criterion, self.valid_data, device)
        for key in valres: res[f'valid_{key}'] = valres[key]

        print(res)
        print(res, file=lf)
        # sys.exit(0)

        with torch.no_grad():
            cs = run_cs()

            mseloss = 0
            for l, r in zip(origpassport, fakepassport):
                mse = F.mse_loss(l, r)
                mseloss += mse.item()
            mseloss /= len(origpassport)

        print(f'MSE of Real and Maximize passport: {mseloss:.4f}')
        print(f'MSE of Real and Maximize passport: {mseloss:.4f}', file=lf)
        print(f'Cosine Similarity of Real and Maximize passport: {sum(cs) / len(origpassport):.4f}')
        print(f'Cosine Similarity of Real and Maximize passport: {sum(cs) / len(origpassport):.4f}', file=lf)
        print()

        res['epoch'] = 0
        res['cosine_similarity'] = cs
        res['flipperc'] = self.flipperc
        res['train_mseloss'] = mseloss

        history.append(res)

        torch.save({'origpassport': origpassport,
                    'fakepassport': fakepassport,
                    'state_dict': self.model.state_dict()},
                   f'{logdir}/{self.arch}-v3-last-{self.dataset}-{self.type}-{self.flipperc:.1f}-e0.pth')

        best_acc = 0
        best_ep = 0
        best_sign_acc = 0

        user_passport_list = []

        for user_idx in range(self.idx):
            user_model_path = f'{basedir}/{self.type}_{str(user_idx + 1)}/user_model.pth'
            user_passport = torch.load(user_model_path)['fakepassport']
            user_passport_list.append(user_passport)

        for ep in range(1, epochs + 1):
            # for m in self.model.modules():
            #     if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
            #         m.key_private.requires_grad = True
            #         m.skey_private.requires_grad = True
            #
            #         for layer in m.fc.modules():
            #             print(layer)
            #             if isinstance(layer, nn.Linear):
            #                 for p in layer.parameters():
            #                     p.requires_grad = True

            # if ep < 10:
            #     for m in self.model.modules():
            #         if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
            #             m.key_private.requires_grad = True
            #             m.skey_private.requires_grad = True
            #
            #             for layer in m.fc.modules():
            #                 if isinstance(layer, nn.Linear):
            #                     for p in layer.parameters():
            #                         p.requires_grad = False
            # else:
            #     for m in self.model.modules():
            #         if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
            #             m.key_private.requires_grad = False
            #             m.skey_private.requires_grad = False
            #
            #             for layer in m.fc.modules():
            #                 if isinstance(layer, nn.Linear):
            #                     for p in layer.parameters():
            #                         p.requires_grad = True

            fcparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'Trainable parameters: {fcparams}')
            trainable_list = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.SGD(trainable_list,
                                        lr=lr,
                                        momentum=0.9,
                                        weight_decay=0.0005)
            # optimizer = torch.optim.Adam(trainable_list, lr=lr, weight_decay=0.0005, betas=(0.9, 0.999))

            # def lr_lambda(epoch):
            #     if 0 <= epoch < 10:
            #         return 1.0
            #     elif 10 <= epoch < 20:
            #         return 0.1
            #     elif 20 <= epoch < 30:
            #         return 1.0
            #     elif 30 <= epoch < 40:
            #         return 0.1
            #     else:
            #         return 0.1

            # scheduler = None
            # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            print(f'Epoch {ep:3d}:')
            print(f'Epoch {ep:3d}:', file=lf)
            print('Training')
            # trainres = train_maximize(origpassport, fakepassport, self.model, optimizer, criterion, self.train_data,
            #                           device, self.type, ep)
            trainres = train_maximize(user_passport_list, origpassport, fakepassport, self.model, optimizer, criterion,
                                      device, self.type, ep)

            print('Testing')
            print('Testing', file=lf)
            valres = test_fake(self.model, criterion, self.valid_data, device)

            print(valres, file=lf)
            print('\n', file=lf)

            if best_sign_acc < valres['signacc']:
                print(f'Found best sign acc at epoch {ep}\n')
                best_sign_acc = valres['signacc']
                best_ep = ep

            f = open(best_file, 'a')
            f.write(str(best_sign_acc) + '\n')
            f.write("best sign epoch: %s" % str(best_ep) + '\n')
            f.flush()

            res = {}

            for key in trainres: res[f'train_{key}'] = trainres[key]
            for key in valres: res[f'valid_{key}'] = valres[key]
            res['epoch'] = ep
            res['flipperc'] = self.flipperc

            with torch.no_grad():
                cs = run_cs()
                res['cosine_similarity'] = cs

            print(f'Cosine Similarity of Real and Maximize passport: '
                  f'{sum(cs) / len(origpassport):.4f}')
            print()

            print(f'Cosine Similarity of Real and Maximize passport: '
                  f'{sum(cs) / len(origpassport):.4f}' + '\n', file=lf)
            lf.flush()

            history.append(res)

            histdf = pd.DataFrame(history)

            if scheduler is not None:
                scheduler.step()

        torch.save({
            'origpassport': origpassport,
            'fakepassport': fakepassport,
            'model': self.model},
            f'{logdir}/user_model.pth'
        )

        histdf.to_csv(f'{logdir}/history.csv')

    #     from .chameleon_hash import owner_chameleon_hash, generate_collision, recover_signature, chameleon_hash
    #
    #     count = 0
    #     for m in passblocks:
    #         print("================================")
    #         print(f'Passport Block {count}')
    #
    #         o = m.conv.out_channels
    #
    #         ori_skey = origpassport[count * 2 + 1].detach()
    #         owner_signature = "Copyright to CVPR 2025"
    #         params, R1, hash1, b = owner_chameleon_hash(ori_skey, owner_signature, hash_length=o)
    #         r1, s1 = R1
    #
    #         skey = m.skey_private.requires_grad_(False)
    #         s2_text = "Authorization to user1"
    #         R2, hash2 = generate_collision(params, hash1, skey, s2_text)
    #         r2, s2 = R2
    #
    #         assert hash1 == hash2, "Collision failed!"
    #         print("Collision successful!")
    #
    #         print(f"hash1: {hash1}\nhash2: {hash2}")
    #
    #         recover_signature(s1)
    #         recover_signature(s2)
    #
    #         count += 1

    def random_attack(self):

        device = self.device

        loadpath = self.logdir
        print("==========================")
        print(loadpath)
        print(self.type)

        task_name = loadpath.split('/')[-2]
        loadpath_all = loadpath + '/models/last.pth'
        sd = torch.load(loadpath_all)

        logdir = ('./passport_attack/' + task_name + '/' + self.experiment_id + '/' + self.type)
        print(logdir)
        os.makedirs(logdir, exist_ok=True)
        shutil.copy('amb_attack.py', str(logdir) + "/amb_attack.py")

        criterion = nn.CrossEntropyLoss()

        acc_list = []

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)

        for i in range(20):
            for m in self.model.modules():
                # print(isinstance(m, PassportPrivateBlock))
                if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                    # if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlockERB):

                    keyname = 'key_private'
                    skeyname = 'skey_private'

                    key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()

                    m.__delattr__(keyname)
                    m.__delattr__(skeyname)

                    noise1 = torch.normal(mean=0, std=1, size=key.size())
                    noise2 = torch.normal(mean=0, std=1, size=skey.size())

                    m.register_parameter(keyname, nn.Parameter(noise1))
                    m.register_parameter(skeyname, nn.Parameter(noise2))

                    # shape = []
                    # x = torch.cuda.FloatTensor(shape)
                    # noise1 = torch.randn(*key.size(), out=x)
                    # noise2 = torch.randn(*key.size(), out=x)

                    # noise1 = torch.randn_like(key)
                    # noise2 = torch.randn_like(skey)

                    # noise1 = torch.zeros_like(key)
                    # noise2 = torch.zeros_like(skey)

                    # m.register_parameter(keyname, nn.Parameter(noise1))
                    # m.register_parameter(skeyname, nn.Parameter(noise2))

            self.model.to(device)
            results = test_fake(self.model, criterion, self.valid_data, device)
            acc = results['acc']
            acc_list.append(acc)

        print(acc_list)

        # save acc_list to file

        with open(logdir + '/acc_list.txt', 'w') as f:
            for item in acc_list:
                f.write("%s\n" % item)
        f.close()

    def check(self):
        def realtime_scale(m, key, skey):
            scale_loss = m.sign_loss_private

            scalekey = m.conv(skey)
            # scalekey = torch.tanh(scalekey)
            b = scalekey.size(0)
            c = scalekey.size(1)

            # scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
            # scale = scale.mean(dim=0).view(1, c, 1, 1)
            scale = scalekey.view(b, c, -1).max(dim=2)[0].view(b, c, 1, 1)
            scale = scale.max(dim=0)[0].view(1, c, 1, 1)

            scale_for_loss = scale
            scale0 = scale.view(1, c)
            # scale = m.fc(scale).view(1, c, 1, 1)

            linear1 = m.fc[0]
            leaky_relu = m.fc[1]
            linear2 = m.fc[2]

            scale1 = linear1(scale0)
            scale2 = leaky_relu(scale1)
            scale3 = linear2(scale2)
            scale = scale3 + scale0

            scale0 = scale0.view(-1).cpu().detach().numpy()
            scale1 = scale1.view(-1).cpu().detach().numpy()
            scale2 = scale2.view(-1).cpu().detach().numpy()
            scale3 = scale3.view(-1).cpu().detach().numpy()
            scale = scale.view(-1).cpu().detach().numpy()

            return scale0, scale1, scale2, scale3, scale

        def realtime_bias(m, key, skey):
            biaskey = m.conv(key)  # key batch always 1
            # biaskey = torch.tanh(biaskey)
            b = biaskey.size(0)
            c = biaskey.size(1)

            # bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
            # bias = bias.mean(dim=0).view(1, c, 1, 1)
            bias = biaskey.view(b, c, -1).max(dim=2)[0].view(b, c, 1, 1)
            bias = bias.max(dim=0)[0].view(1, c, 1, 1)

            bias0 = bias.view(1, c)
            # bias = m.fc(bias).view(1, c, 1, 1)

            linear1 = m.fc[0]
            leaky_relu = m.fc[1]
            linear2 = m.fc[2]

            bias1 = linear1(bias0)
            bias2 = leaky_relu(bias1)
            bias3 = linear2(bias2)
            bias = bias3 + bias0

            bias0 = bias0.view(-1).cpu().detach().numpy()
            bias1 = bias1.view(-1).cpu().detach().numpy()
            bias2 = bias2.view(-1).cpu().detach().numpy()
            bias3 = bias3.view(-1).cpu().detach().numpy()
            bias = bias.view(-1).cpu().detach().numpy()

            return bias0, bias1, bias2, bias3, bias

        device = self.device
        loadpath = self.logdir
        task_name = loadpath.split('/')[-2]
        basedir = './passport_attack/' + task_name + '/'


        acc_array = np.zeros((5, 5))
        cs_arracy = np.zeros((5, 5))

        for i in range(5):
            for j in range(5):

                # owner_path = "./results/alexnet_cifar10_v4_last1/4_bn_image/models/last.pth"
                # attack_path = "./results/resnet_cifar10_v4_l4/4_bn_image/models/last.pth"
                owner_path = f'{basedir}/oracle_{str(i + 1)}/user_model.pth'
                attack_path = f'{basedir}/oracle_{str(j + 1)}/user_model.pth'

                # print(owner_path)
                # print(attack_path)

                # passport_kwargs = construct_passport_kwargs(self)
                #
                # if self.arch == 'alexnet':
                #     model_1 = AlexNetPassportPrivate(self.in_channels, self.num_classes, passport_kwargs)
                #     model_2 = AlexNetPassportPrivate(self.in_channels, self.num_classes, passport_kwargs)
                # else:
                #     model_1 = ResNet18Private(num_classes=self.num_classes, passport_kwargs=passport_kwargs)
                #     model_2 = ResNet18Private(num_classes=self.num_classes, passport_kwargs=passport_kwargs)

                # sd = torch.load(owner_path)
                # model_1.load_state_dict(sd, strict=False)
                # self.model.to(self.device)

                # sd = torch.load(attack_path)
                # model_2.load_state_dict(sd, strict=False)
                # self.model.to(self.device)

                model_1 = torch.load(owner_path)['model']
                model_2 = torch.load(attack_path)['model']

                model_1.to(device)
                model_2.to(device)

                key_1_list = []
                skey_1_list = []

                for m in model_1.modules():
                    # print(isinstance(m, PassportPrivateBlock))
                    if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                        # if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlockERB):

                        keyname = 'key_private'
                        skeyname = 'skey_private'

                        key = m.__getattr__(keyname).data.clone()
                        skey = m.__getattr__(skeyname).data.clone()

                        key_1_list.append(key.to(device))
                        skey_1_list.append(skey.to(device))

                key_2_list = []
                skey_2_list = []

                for m in model_2.modules():
                    # print(isinstance(m, PassportPrivateBlock))
                    if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                        # if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlockERB):

                        keyname = 'key_private'
                        skeyname = 'skey_private'

                        key = m.__getattr__(keyname).data.clone()
                        skey = m.__getattr__(skeyname).data.clone()

                        print(key.min(), key.max())
                        print(skey.min(), skey.max())

                        key_2_list.append(key.to(device))
                        skey_2_list.append(skey.to(device))

                cs_criterion = nn.CosineSimilarity()
                cs = 0
                for l, r in zip(skey_1_list, skey_2_list):
                    cs += cs_criterion(l.view(1, -1), r.view(1, -1)).mean().item()

                cs /= len(skey_1_list)

                # old_scale_list_0 = []
                # old_scale_list_1 = []
                # old_scale_list_2 = []
                # old_scale_list_3 = []
                # old_scale_list = []
                # old_bias_list_0 = []
                # old_bias_list_1 = []
                # old_bias_list_2 = []
                # old_bias_list_3 = []
                # old_bias_list = []
                #
                # new_scale_list_0 = []
                # new_scale_list_1 = []
                # new_scale_list_2 = []
                # new_scale_list_3 = []
                # new_scale_list = []
                # new_bias_list_0 = []
                # new_bias_list_1 = []
                # new_bias_list_2 = []
                # new_bias_list_3 = []
                # new_bias_list = []
                #
                # count = 0
                #
                # for m in model_1.modules():
                #     # print(isinstance(m, PassportPrivateBlock))
                #     if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                #         # if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlockERB):
                #
                #         # _, old_scale = realtime_scale(m, key_1_list[count], skey_1_list[count])
                #         o_scale0, o_scale1, o_scale2, o_scale3, o_scale = realtime_scale(m, key_1_list[count],
                #                                                                          skey_1_list[count])
                #         o_bias0, o_bias1, o_bias2, o_bias3, o_bias = realtime_bias(m, key_1_list[count], skey_1_list[count])
                #
                #         # old_scale_list.append(old_scale)
                #         old_scale_list_0.append(o_scale0)
                #         old_scale_list_1.append(o_scale1)
                #         old_scale_list_2.append(o_scale2)
                #         old_scale_list_3.append(o_scale3)
                #         old_scale_list.append(o_scale)
                #         old_bias_list_0.append(o_bias0)
                #         old_bias_list_1.append(o_bias1)
                #         old_bias_list_2.append(o_bias2)
                #         old_bias_list_3.append(o_bias3)
                #         old_bias_list.append(o_bias)
                #
                #         count += 1
                #
                # count = 0
                #
                # for m in model_1.modules():
                #     if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                #         # _, new_scale = realtime_scale(m, key_2_list[count], skey_2_list[count])
                #         n_scale0, n_scale1, n_scale2, n_scale3, n_scale = realtime_scale(m, key_2_list[count],
                #                                                                          skey_2_list[count])
                #         n_bias0, n_bias1, n_bias2, n_bias3, n_bias = realtime_bias(m, key_2_list[count], skey_2_list[count])
                #
                #         # new_scale_list.append(new_scale)
                #         new_scale_list_0.append(n_scale0)
                #         new_scale_list_1.append(n_scale1)
                #         new_scale_list_2.append(n_scale2)
                #         new_scale_list_3.append(n_scale3)
                #         new_scale_list.append(n_scale)
                #         new_bias_list_0.append(n_bias0)
                #         new_bias_list_1.append(n_bias1)
                #         new_bias_list_2.append(n_bias2)
                #         new_bias_list_3.append(n_bias3)
                #         new_bias_list.append(n_bias)
                #
                #         count += 1
                #
                # o_scale0 = old_scale_list_0[-1]
                # o_scale1 = old_scale_list_1[-1]
                # o_scale2 = old_scale_list_2[-1]
                # o_scale3 = old_scale_list_3[-1]
                # o_scale = old_scale_list[-1]
                #
                # n_scale0 = new_scale_list_0[-1]
                # n_scale1 = new_scale_list_1[-1]
                # n_scale2 = new_scale_list_2[-1]
                # n_scale3 = new_scale_list_3[-1]
                # n_scale = new_scale_list[-1]
                #
                # scale0_data = [o_scale0, n_scale0]
                # scale1_data = [o_scale1, n_scale1]
                # scale2_data = [o_scale2, n_scale2]
                # scale3_data = [o_scale3, n_scale3]
                # scale_data = [o_scale, n_scale]
                #
                # o_bias0 = old_bias_list_0[-1]
                # o_bias1 = old_bias_list_1[-1]
                # o_bias2 = old_bias_list_2[-1]
                # o_bias3 = old_bias_list_3[-1]
                # o_bias = old_bias_list[-1]
                #
                # n_bias0 = new_bias_list_0[-1]
                # n_bias1 = new_bias_list_1[-1]
                # n_bias2 = new_bias_list_2[-1]
                # n_bias3 = new_bias_list_3[-1]
                # n_bias = new_bias_list[-1]
                #
                # bias0_data = [o_bias0, n_bias0]
                # bias1_data = [o_bias1, n_bias1]
                # bias2_data = [o_bias2, n_bias2]
                # bias3_data = [o_bias3, n_bias3]
                # bias_data = [o_bias, n_bias]
                #
                # skey_data = [skey_1_list[-1].view(-1).cpu().detach().numpy(), skey_2_list[-1].view(-1).cpu().detach().numpy()]
                # key_data = [key_1_list[-1].view(-1).cpu().detach().numpy(), key_2_list[-1].view(-1).cpu().detach().numpy()]
                #
                # fig, ax = plt.subplots(2, 6, figsize=(20, 10))
                #
                # ax[0, 0].violinplot(skey_data)
                # ax[0, 0].set_title('Skey')
                #
                # ax[0, 1].violinplot(scale0_data)
                # ax[0, 1].set_title('Scale 0')
                #
                # ax[0, 2].violinplot(scale1_data)
                # ax[0, 2].set_title('Scale 1')
                #
                # ax[0, 3].violinplot(scale2_data)
                # ax[0, 3].set_title('Scale 2')
                #
                # ax[0, 4].violinplot(scale3_data)
                # ax[0, 4].set_title('Scale 3')
                #
                # ax[0, 5].violinplot(scale_data)
                # ax[0, 5].set_title('Scale')
                #
                # ax[1, 0].violinplot(key_data)
                # ax[1, 0].set_title('Key')
                #
                # ax[1, 1].violinplot(bias0_data)
                # ax[1, 1].set_title('Bias 0')
                #
                # ax[1, 2].violinplot(bias1_data)
                # ax[1, 2].set_title('Bias 1')
                #
                # ax[1, 3].violinplot(bias2_data)
                # ax[1, 3].set_title('Bias 2')
                #
                # ax[1, 4].violinplot(bias3_data)
                # ax[1, 4].set_title('Bias 3')
                #
                # ax[1, 5].violinplot(bias_data)
                # ax[1, 5].set_title('Bias')
                #
                # plt.show()

                count = 0
                for m in model_1.modules():
                    # print(isinstance(m, PassportPrivateBlock))
                    if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
                        # if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlockERB):

                        keyname = 'key_private'
                        skeyname = 'skey_private'

                        m.__delattr__(keyname)
                        m.__delattr__(skeyname)

                        m.register_parameter(keyname, nn.Parameter(key_2_list[count]))
                        m.register_parameter(skeyname, nn.Parameter(skey_2_list[count]))

                        count += 1

                model_1.to(device)
                model_2.to(device)

                criterion = nn.CrossEntropyLoss()
                results = test_fake(model_1, criterion, self.valid_data, device)
                acc = results['acc']

                acc_array[i][j] = acc
                cs_arracy[i][j] = cs

        print(acc_array)
        print(cs_arracy)

        # np.save('acc_array.npy', acc_array)
        # np.save('cs_array.npy', cs_arracy)

        # print("*************************************")
        #
        # from .chameleon_hash import owner_chameleon_hash, generate_collision, recover_license
        #
        # count = 0
        # for m in fakepassblocks:
        #     print("================================")
        #     print(f'Passport Block {count}')
        #
        #     o = m.conv.out_channels
        #     signature = m.b.cpu()
        #
        #     ori_skey = origpassport[count].detach()
        #     # ori_skey = torch.randn_like(ori_skey)
        #     owner_license = "Copyright to CVPR 2025"
        #     params, r1, hash1, b = owner_chameleon_hash(ori_skey, owner_license, hash_length=o)
        #
        #     recover_license(r1)
        #
        #     signature = (signature + 1) * 0.5
        #     b = (b + 1) * 0.5
        #
        #     error_rate = torch.sum(torch.abs(signature - b)) / b.size(0)
        #     print(error_rate)
        #
        #     # sign_acc = m.sign_loss_private.b.cpu()
        #     # print(sign_acc)
        #
        #     count += 1