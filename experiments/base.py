import csv
import json
import os

import torch
import collections


class Experiment(object):
    """
    1. load variables
    2. load dataset
    3. load model
    4. load optimizer
    5. load trainer
    6. self.makedirs_or_load(args)
    """

    def __init__(self, args):
        self.args = args
        self.model = None
        self.prefix = ''
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.experiment_id = args['exp_id']
        # self.run_id = args['run_id']
        self.attack = args['attack']
        self.flipperc = args['flipperc']
        self.type = args['type']  # key type

        self.buffer = []
        self.save_history_interval = 1
        self.device = torch.device('cuda')

        self.arch = args['arch']
        self.dataset = args['dataset']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.eval = args['eval']
        self.tag = args['tag']
        self.save_interval = args['save_interval']
        self.lr_config = json.load(open(args['lr_config']))
        self.pretrained_path = args['pretrained_path']

        self.norm_type = args['norm_type']
        self.train_backdoor = args['train_backdoor']

        self.train_passport = args['train_passport']  # v1 DeepIPR
        self.train_private = args['train_private']  # v2 Passport-aware Normalization
        self.hash = args['hash']  # v3 Trapdoor Normalization
        self.chameleon = args['chameleon']  # v4 Chameleon passport (Ours)

        if self.train_passport:
            self.scheme = 1
            print('DeepIPR Scheme')
        elif self.train_private:
            if not self.hash:
                self.scheme = 2
                print('Passport-aware Normalization Scheme')
            else:
                if not self.chameleon:
                    self.scheme = 3
                    print('Trapdoor Normalization Scheme')
                else:
                    self.scheme = 4
                    print('Chameleon Passport Scheme')
        else:  # baseline
            self.scheme = 0

        self.passport_config = json.load(open(args['passport_config']))

        self.sl_ratio = args['sign_loss']
        self.key_type = args['key_type']
        self.use_trigger_as_passport = args['use_trigger_as_passport']

        self.is_tl = args['transfer_learning']
        self.tl_dataset = args['tl_dataset']
        self.tl_scheme = args['tl_scheme']

        self.logdir = f'/mnt/UserData1/chaohui/Projects/CHIP/Image_cls/Ours/results/{self.arch}_{self.dataset}'

        self.logdir += f'_v{self.scheme}'

        if self.tag is not None:
            self.logdir += f'_{self.tag}'

    def get_expid(self, logdir, prefix):
        exps = [d.replace(prefix, '') for d in os.listdir(logdir) if
                os.path.isdir(os.path.join(logdir, d)) and prefix in d]
        files = set(map(int, exps))
        if len(files):
            return min(set(range(1, max(files) + 2)) - files)
        else:
            return 1

    def finetune_load(self):
        # create directory like this: logdir/tl_{expid}

        self.prefix = 'tl_' + self.tl_dataset + '_'
        self.logdir = os.path.join(self.logdir, str(self.experiment_id))

        # path = os.path.join(self.logdir, 'models', 'best.pth')
        path = os.path.join(self.logdir, 'models', 'last.pth')
        if not os.path.exists(path):
            print(f'Warning: No such Experiment -> {path}')
        else:
            # self.load_model('best.pth')
            self.load_model('last.pth')

        self.finetune_id = self.get_expid(self.logdir, self.prefix)

        self.logdir = os.path.join(self.logdir, f'{self.prefix}{self.finetune_id}')

        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(os.path.join(self.logdir, 'models'), exist_ok=True)

        json.dump(self.args, open(os.path.join(self.logdir, 'config.json'), 'w'), indent=4)
        self.model = self.model.to(self.device)

    def makedirs_or_load(self):
        # create directory like this: logdir/{expid}, expid + 1 if exist

        os.makedirs(self.logdir, exist_ok=True)

        if not self.eval:
            # create experiment directory
            # if self.attack:
            #     self.experiment_id = 1
            # else:
            # self.experiment_id = self.get_expid(self.logdir, self.prefix)

            self.logdir = os.path.join(self.logdir, str(self.experiment_id))

            # create sub directory
            os.makedirs(os.path.join(self.logdir, 'models'), exist_ok=True)

            # write config
            json.dump(self.args, open(os.path.join(self.logdir, 'config.json'), 'w'), indent=4)
        else:
            self.experiment_id = self.args['exp_id']
            self.logdir = os.path.join(self.logdir, str(self.args['exp_id']))
            path = os.path.join(self.logdir, 'models', 'best.pth')

            # check experiment exists
            if not os.path.exists(path):
                print(f'Warning: No such Experiment -> {path}')
            else:
                self.load_model('best.pth')

            self.model = self.model.to(self.device)

    def save_model(self, filename, model=None):
        if model is None:
            model = self.model

        torch.save(model.cpu().state_dict(), os.path.join(self.logdir, f'models/{filename}'))
        model.to(self.device)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.logdir, f'models/{filename}')))

    def ft_state_dict(self, filename):
        old_state = torch.load(os.path.join(self.logdir, f'models/{filename}'))
        new_state_dict = collections.OrderedDict()
        for k, v in old_state.items():
            new_state_dict[k.replace('bn0', 'bn')] = v
        # self.ft_model.load_state_dict(new_state_dict)
        return new_state_dict

    def save_last_model(self, model=None):
        self.save_model('last.pth', model)

    def training(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def flush_history(self, history_file, first):
        if len(self.buffer) != 0:
            columns = sorted(self.buffer[0].keys())
            with open(history_file, 'a') as file:
                writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
                if first:
                    writer.writerow(columns)

                for data in self.buffer:
                    writer.writerow(list(map(lambda x: data[x], columns)))

            self.buffer.clear()

    def append_history(self, history_file, data, first=False):  # row by row
        self.buffer.append(data)

        if len(self.buffer) >= self.save_history_interval:
            self.flush_history(history_file, first)
