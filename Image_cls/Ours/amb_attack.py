import json
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.resnet_passport_private import ResNetPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.losses.sign_loss import SignLoss
import shutil

class DatasetArgs():
    pass

def train_maximize(
        user_passport_list,
        origpassport, fakepassport, model, optimizer, criterion, trainloader, device, type, ep
):
    model.train()
    loss_meter = 0
    signloss_meter = 0
    maximizeloss_meter = 0
    mseloss_meter = 0
    csloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    balloss_meter = 0
    factorloss_meter = 0
    l2norm_meter = 0
    start_time = time.time()
    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()   #?????

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
        # scale = scale.view(1, c)
        # scale = m.fc(scale).view(1, c, 1, 1)

        if scale_loss is not None:
            scale_loss.reset()
            # scale_loss.add(scale)
            scale_loss.add(scale_for_loss)

        return scale_for_loss, scale

    def realtime_bias(m, key, skey):
        biaskey = m.conv(key)  # key batch always 1
        # biaskey = torch.tanh(biaskey)
        b = biaskey.size(0)
        c = biaskey.size(1)

        # bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        # bias = bias.mean(dim=0).view(1, c, 1, 1)
        bias = biaskey.view(b, c, -1).max(dim=2)[0].view(b, c, 1, 1)
        bias = bias.max(dim=0)[0].view(1, c, 1, 1)

        # bias = bias.view(1, c)
        # bias = m.fc(bias).view(1, c, 1, 1)

        return bias

    num_batch = len(trainloader)

    base_passport = fakepassport.copy()

    for k, (d, t) in enumerate(trainloader):
        # if k <= num_batch:
        # if k <= int(num_batch / 10):
        if k <= 100:

            optimizer.zero_grad()

            # d = d.to(device)
            # t = t.to(device)
            # # if scheme == 1:
            # #     pred = model(d)
            # # else:
            # pred = model(d, ind=1)  #private graph
            #
            # loss = criterion(pred, t)

            loss = torch.tensor(0.).to(device)

            balloss = torch.tensor(0.).to(device)

            for m in model.modules():
                if isinstance(m, PassportPrivateBlock):
                    old_scale = m.old_scale
                    old_bias = m.old_bias

                    _, new_scale = m.get_scale_private()
                    new_bias = m.get_bias_private()

                    old = torch.cat([old_scale.view(-1), old_bias.view(-1)])
                    new = torch.cat([new_scale.view(-1), new_bias.view(-1)])

                    balloss += mse_criterion(old, new)

            factorloss = torch.tensor(0.).to(device)

            # count = 0
            # for m in model.modules():
            #     if isinstance(m, PassportPrivateBlock):
            #         # user1_key = user1_passport[count]
            #         # user1_skey = user1_passport[count + 1]
            #
            #         fake_key = fakepassport[count]
            #         fake_skey = fakepassport[count + 1]
            #
            #         noise1 = torch.randn_like(fake_key) * 0.1
            #         noise2 = torch.randn_like(fake_skey) * 0.1
            #
            #         user1_key = noise1 + fake_key
            #         user1_skey = noise2 + fake_skey
            #
            #         # print("user1_key:", user1_key.size())
            #         # print("user1_skey:", user1_skey.size())
            #         # print("fake_key:", fake_key.size())
            #         # print("fake_skey:", fake_skey.size())
            #
            #         _, user_scale = realtime_scale(m, user1_key, user1_skey)
            #         _, fake_scale = realtime_scale(m, fake_key, fake_skey)
            #
            #         user_bias = realtime_bias(m, user1_key, user1_skey)
            #         fake_bias = realtime_bias(m, fake_key, fake_skey)
            #
            #         factorloss += 1 / (mse_criterion(user_scale, fake_scale) + 0.0001)
            #         factorloss += 1 / (mse_criterion(user_bias, fake_bias) + 0.0001)
            #
            #         count += 2

            signloss = torch.tensor(0.).to(device)
            signacc = torch.tensor(0.).to(device)
            count = 0

            for m in model.modules():
                if isinstance(m, SignLoss):
                    signloss += m.loss
                    signacc += m.acc
                    count += 1

            l2_norm = torch.tensor(0.).to(device)
            trainable_list = [p for p in model.parameters() if p.requires_grad]
            l2_norm += sum([torch.norm(p) ** 2 for p in trainable_list])
            l2_norm /= len(trainable_list)

            maximizeloss = torch.tensor(0.).to(device)
            mseloss = torch.tensor(0.).to(device)
            csloss = torch.tensor(0.).to(device)

            for l, r in zip(origpassport, fakepassport):
                mse = mse_criterion(l, r)
                cs = cs_criterion(l.view(1, -1), r.view(1, -1)).mean()
                csloss += cs
                mseloss += mse
                maximizeloss += 1 / mse

            for user_passport in user_passport_list:
                for l, r in zip(user_passport, fakepassport):
                    mse = mse_criterion(l, r)
                    cs = cs_criterion(l.view(1, -1), r.view(1, -1)).mean()
                    csloss += cs
                    mseloss += mse
                    maximizeloss += 1 / mse

            if 'fake2-' in type  :
                (balloss).backward()  #only cross-entropy loss  backward  fake2
            elif  'fake3-' in type :
                (balloss + maximizeloss).backward()  #csloss do not backward   kafe3

            else:
                # print("FFFFFFFFFFFFFFFFFF")

                # if ep < 10:
                #     (maximizeloss + signloss).backward()
                #     torch.nn.utils.clip_grad_norm_(fakepassport, 20)  # ????
                # else:
                #     (balloss).backward()

                (balloss + signloss + maximizeloss).backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)  # ????

                optimizer.step()

                # for i in range(len(fakepassport)):
                #     fakepassport[i] = torch.clamp(fakepassport[i], base_passport[i] - 0.5, base_passport[i] + 0.5)

                # acc = (pred.max(dim=1)[1] == t).float().mean()
                acc = torch.tensor(0.).to(device)

                loss_meter += loss.item()
                acc_meter += acc.item()
                signloss_meter += signloss.item()
                signacc_meter += signacc.item() / count
                maximizeloss_meter += maximizeloss.item()
                mseloss_meter += mseloss.item()
                csloss_meter += csloss.item()
                balloss_meter += balloss.item()
                l2norm_meter += l2_norm.item()
                factorloss_meter += factorloss.item()

                tqdm.write(f'Batch [{k + 1}/{len(trainloader)}]: '
                      f'Loss: {loss_meter / (k + 1):.4f} '
                      f'Acc: {acc_meter / (k + 1):.4f} '
                      f'Sign Loss: {signloss_meter / (k + 1):.4f} '
                      f'Sign Acc: {signacc_meter / (k + 1):.4f} '
                      f'MSE Loss: {mseloss_meter / (k + 1):.4f} '
                      f'Maximize Dist: {maximizeloss_meter / (k + 1):.4f} '
                      f'Bal Loss: {balloss_meter / (k + 1):.4f} '
                      f'L2 Norm: {l2norm_meter / (k + 1):.4f} '
                      f'CS: {csloss_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)'
                      )

    print()
    loss_meter /= len(trainloader)
    acc_meter /= len(trainloader)
    signloss_meter /= len(trainloader)
    signacc_meter /= len(trainloader)
    maximizeloss_meter /= len(trainloader)
    mseloss_meter /= len(trainloader)
    csloss_meter /= len(trainloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
            'maximizeloss': maximizeloss_meter,
            'mseloss': mseloss_meter,
            'csloss': csloss_meter,
            'time': start_time - time.time()}

def train_ERB(model, optimizer, criterion, trainloader, device, type, ep):
    model.train()
    loss_meter = 0
    signloss_meter = 0
    balloss_meter = 0
    maximizeloss_meter = 0
    mseloss_meter = 0
    csloss_meter = 0
    dep_acc_meter = 0
    fore_acc_meter = 0
    signacc_meter = 0
    start_time = time.time()
    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()
    for k, (d, t) in enumerate(trainloader):
        d = d.to(device)
        t = t.to(device)
        for m in model.modules():
            if isinstance(m, SignLoss):
                m.reset()

        loss = torch.tensor(0.).to(device)
        signloss = torch.tensor(0.).to(device)
        balloss = torch.tensor(0.).to(device)
        signacc = torch.tensor(0.).to(device)
        count, count_ = 0, 0

        pred = model(d, ind=0)
        loss += criterion(pred, t)
        acc = (pred.max(dim=1)[1] == t).float().mean()
        dep_acc_meter += acc.item()

        pred = model(d, ind=1)
        loss += criterion(pred, t)
        acc = (pred.max(dim=1)[1] == t).float().mean()
        fore_acc_meter += acc.item()

        # sign loss
        for m in model.modules():
            if isinstance(m, SignLoss):
                signloss += m.loss
                signacc += m.acc
                count += 1

        # for m in model.modules():
        #     if isinstance(m, PassportPrivateBlock):
        #         balloss += m.get_loss()
        #         count_ += 1

        (loss + signloss).backward()
        optimizer.step()

        loss_meter += loss.item()
        signloss_meter += signloss.item()
        #balloss_meter += balloss.item() / count_
        signacc_meter += signacc.item() / count

        print(f'Batch [{k + 1}/{len(trainloader)}]: '
              f'Loss: {loss_meter / (k + 1):.2f} '
              f'Fore. Acc: {100*fore_acc_meter / (k + 1):.2f} '
              f'Dep. Acc: {100*dep_acc_meter / (k + 1):.2f} '
              #f'Bal Loss: {balloss_meter / (k + 1):.2f} '
              f'Bal Dis: { 100*(np.abs(dep_acc_meter-fore_acc_meter)) / (k + 1):.2f} '
              f'Sign Loss: {signloss_meter / (k + 1):.2f} '
              f'Sign Acc: {100*signacc_meter / (k + 1):.2f} '
              ,
              end='\r')
        # if ep == 1:
        #     wandb.log({
        #             "AMB_ERB_training/Training loss": loss_meter / (k + 1) if loss_meter / (k + 1)<=200 else np.nan,
        #             "AMB_ERB_training/Sign Loss": signloss_meter / (k + 1) if signloss_meter / (k + 1)<=200 else np.nan,
        #             "AMB_ERB_training/Dep. acc": 100*dep_acc_meter / (k + 1),
        #             "AMB_ERB_training/Fore. acc": 100*fore_acc_meter / (k + 1),
        #             })

    print()
    loss_meter /= len(trainloader)
    fore_acc_meter /= len(trainloader)
    dep_acc_meter /= len(trainloader)
    signloss_meter /= len(trainloader)
    #balloss_meter /= len(trainloader)
    signacc_meter /= len(trainloader)


    return {'loss': loss_meter,
            'signloss': signloss_meter,
            #'balloss': balloss_meter,
            'depacc': dep_acc_meter,
            'foreacc': fore_acc_meter,
            'signacc': signacc_meter,
            'baldis': np.abs(dep_acc_meter-fore_acc_meter),
            'time': start_time - time.time()}


def test_fake(model, criterion, valloader, device):
    model.eval()
    loss_meter = 0
    signloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    start_time = time.time()

    with torch.no_grad():
        for k, (d, t) in enumerate(valloader):
            d = d.to(device)
            t = t.to(device)

            # if scheme == 1:
            #     pred = model(d)
            # else:
            pred = model(d, ind=1)

            loss = criterion(pred, t)

            signloss = torch.tensor(0.).to(device)
            signacc = torch.tensor(0.).to(device)
            count = 0

            for m in model.modules():
                if isinstance(m, SignLoss):
                    signloss += m.get_loss()
                    signacc += m.get_acc()
                    count += 1

            acc = (pred.max(dim=1)[1] == t).float().mean()

            loss_meter += loss.item()
            acc_meter += acc.item()
            signloss_meter += signloss.item()
            try:
                signacc_meter += signacc.item() / count
            except:
                pass

            print(f'Batch [{k + 1}/{len(valloader)}]: '
                  f'Loss: {loss_meter / (k + 1):.4f} '
                  f'Acc: {acc_meter / (k + 1):.4f} '
                  f'Sign Loss: {signloss_meter / (k + 1):.4f} '
                  f'Sign Acc: {signacc_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
                  end='\r')

    print()

    loss_meter /= len(valloader)
    acc_meter /= len(valloader)
    signloss_meter /= len(valloader)
    signacc_meter /= len(valloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
            'time': time.time() - start_time}



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='fake attack 3: create another passport maximized from current passport')
    # parser.add_argument('--rep', default=1, type=int,  help='training id')
    parser.add_argument('--rep', default=1, type=str,  help='training comment')
    parser.add_argument('--arch', default='resnet18', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--flipperc', default=0, type=float,
                        help='flip percentange 0~1')
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    args = parser.parse_args()

    args.scheme = 3
    args.loadpath = "/data-x/g12/zhangjie/DeepIPR/baseline/resnet18_cifar100_v3_all/"
    # print(args.loadpath.split('/')[-2])
    # sys.exit(0)
    print(args.loadpath)