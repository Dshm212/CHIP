import errno
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader, make_dataset, IMG_EXTENSIONS
from torchvision.transforms import transforms


class Caltech101(Dataset):
    link = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
    filename = '101_ObjectCategories.tar.gz'
    foldername = '101_ObjectCategories'

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        # classes.remove('BACKGROUND_Google')

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __init__(self, root, train=True, transform=None, download=True):
        self.root = root
        root = os.path.join(root, self.foldername)

        if download:
            self.download()

        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)

        datapaths = defaultdict(list)

        for path, target in samples:
            datapaths[target].append(path)

        for target in datapaths.keys():
            if train:
                datapaths[target] = datapaths[target][:int(0.8 * len(datapaths[target]))]
            else:
                datapaths[target] = datapaths[target][int(0.8 * len(datapaths[target])):]

        newdatapaths = []
        labels = []
        for target in datapaths.keys():
            for path in datapaths[target]:
                newdatapaths.append(path)
                labels.append(target)

        self.train = train
        self.transform = transform
        self.labels = labels
        self.datapaths = newdatapaths
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            self.cache[index] = img

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, self.filename)):
            print('Files already downloaded and verified')
            return

        self.download_url(self.link)

        # extract file
        cwd = os.getcwd()
        mode = 'r:gz' if self.filename.endswith('.gz') else 'r'
        tar = tarfile.open(os.path.join(self.root, self.filename), mode)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def download_url(self, link):
        from six.moves import urllib

        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # downloads file
        try:
            print('Downloading ' + link + ' to ' + fpath)
            urllib.request.urlretrieve(link, fpath)
        except:
            if link[:5] == 'https':
                url = link.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

    def __len__(self):
        return len(self.datapaths)


class Caltech256(Caltech101):
    link = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
    filename = '256_ObjectCategories.tar'
    foldername = '256_ObjectCategories'


class WMDataset(Dataset):
    def __init__(self, root, labelpath, transform):
        self.root = root

        self.datapaths = [os.path.join(self.root, fn) for fn in os.listdir(self.root)]
        self.labelpath = labelpath
        self.labels = np.loadtxt(self.labelpath)
        self.transform = transform
        self.cache = {}

    def __getitem__(self, index):
        target = self.labels[index]
        if index in self.cache:
            img = self.cache[index]
        else:
            path = self.datapaths[index]
            img = pil_loader(path)
            img = self.transform(img)  # transform is fixed CenterCrop + ToTensor
            self.cache[index] = img

        return img, int(target)

    def __len__(self):
        return len(self.datapaths)


def prepare_wm(datapath='data/trigger_set/pics',shuffle=True):
    triggerroot = datapath
    labelpath = 'data/trigger_set/labels-cifar.txt'

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_list = [
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ]

    transform_list.append(transforms.Normalize(mean, std))

    wm_transform = transforms.Compose(transform_list)

    dataset = WMDataset(triggerroot, labelpath, wm_transform)

    loader = DataLoader(dataset,
                        batch_size=2,
                        shuffle=shuffle,
                        drop_last=True)

    return loader


def prepare_dataset(args):
    is_tl = args['transfer_learning']
    tl_ds = args['tl_dataset']
    ds = args['dataset'] if not is_tl else tl_ds

    ##### shortcut ######
    is_cifar = 'cifar' in ds
    root = f'data/{ds}'
    print('Loading dataset: ' + ds)

    DATASET = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'caltech-101': Caltech101,
        'caltech-256': Caltech256
    }[ds]

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    ##### train transform #####
    if not is_cifar:
        transform_list = [
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ]
    else:
        transform_list = []

    transform_list.extend([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_transforms = transforms.Compose(transform_list)

    ##### test transform #####
    if not is_cifar:
        transform_list = [
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ]
    else:
        transform_list = []

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose(transform_list)

    ##### dataset and loader #####
    train_dataset = DATASET(root,
                            train=True,
                            transform=train_transforms,
                            download=True)
    test_dataset = DATASET(root,
                           train=False,
                           transform=test_transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args['batch_size'] * 2,
                             shuffle=False,
                             num_workers=4)

    return train_loader, test_loader


def prepare_ERB_dataset(args):
    ds = args['dataset']
    ##### shortcut ######
    is_cifar = 'cifar' in ds
    root = f'./data/{ds}'
    print('Loading dataset: ' + ds)

    DATASET = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'caltech-101': Caltech101,
        'caltech-256': Caltech256
    }[ds]

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    ##### train transform #####
    if not is_cifar:
        transform_list = [
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ]
    else:
        transform_list = []

    transform_list.extend([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_transforms = transforms.Compose(transform_list)

    ##### test transform #####
    if not is_cifar:
        transform_list = [
            transforms.Resize(32),
            transforms.CenterCrop(32)
        ]
    else:
        transform_list = []

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose(transform_list)

    ##### dataset and loader #####
    train_dataset = DATASET(root,
                            train=True,
                            transform=train_transforms,
                            download=True)
    test_dataset = DATASET(root,
                           train=False,
                           transform=test_transforms)

    random.seed(42)
    selected_indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * 0.1))
    subset_dataset = Subset(train_dataset, selected_indices)

    train_loader = DataLoader(subset_dataset,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args['batch_size'] * 1,
                             shuffle=False,
                             num_workers=0)

    return train_loader, test_loader


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt, transform):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0])
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        # ten_class = [139, 263, 281, 339, 346, 348, 404, 705, 817, 907]
        # img = []
        fn = self.imgs[index]
        fn = '/mnt/ssd4/chaohui/datasets/ImageNet/raw_data/val/ILSVRC2012_img_val/' + fn
        img = Image.open(fn).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)


def prepare_attack_dataset(rd=42):
    val_txt = '/mnt/ssd4/chaohui/datasets/ImageNet/raw_data/val.txt'
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    coco_dataset = MyDataset(
        txt=val_txt,
        transform=transform
    )
    random.seed(rd)
    selected_indices = random.sample(range(len(coco_dataset)), 200)
    subset_dataset = Subset(coco_dataset, selected_indices)
    batch_size = 1
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    return dataloader