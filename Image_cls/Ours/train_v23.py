import argparse
from pprint import pprint

from experiments.classification_private import ClassificationPrivateExperiment
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet'],
                        help='architecture (default: alexnet)')
    parser.add_argument('--passport-config', default='passport_configs/alexnet_passport_3.json',
                        help='should be same json file as arch')

    parser.add_argument('-d', '--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100', 'caltech-101', 'caltech-256'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--tl-dataset', default='cifar100',
                        choices=['cifar10', 'cifar100','caltech-101', 'caltech-256'],
                        help='transfer learning dataset (default: cifar100)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')

    parser.add_argument('-n', '--norm-type', default='bn', choices=['bn', 'gn', 'in', 's_bn', 'nose'],
                        help='norm type (default: bn)')
    parser.add_argument('-t', '--tag', help='tag')
    # parser.add_argument('-t', '--tag', default='456-our', help='tag')  #only 456
    # parser.add_argument('-t', '--tag', default='4-our', help='tag')  #only layer 4

    # passport argument
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epochs (default: 200)')
    parser.add_argument('--key-type', choices=['random', 'image', 'shuffle'], default='random',
                        help='passport key type (default: shuffle)')
    parser.add_argument('-s', '--sign-loss', type=float, default=0.1,
                        help='sign loss to avoid scale not trainable (default: 0.1)')
    parser.add_argument('--use-trigger-as-passport', action='store_true', default=False,
                        help='use trigger data as passport')

    parser.add_argument('--train-passport', action='store_true', default=False,
                        help='train passport')
    parser.add_argument('-tb', '--train-backdoor', action='store_true', default=False,
                        help='train backdoor')
    parser.add_argument('--train-private', action='store_true', default=False,
                        help='train private')
    parser.add_argument('--hash', action='store_true', default=False,
                        help='whether to use hash to generate the signature')
    parser.add_argument('--chameleon', action='store_true', default=False,
                        help='whether to use chameleon hash')

    parser.add_argument('--attack', default=False,
                        help='attack the pretrained model')
    # paths
    parser.add_argument('--pretrained-path',
                        help='load pretrained path')
    parser.add_argument('--lr-config', default='lr_configs/default.json',
                        help='lr config json file')


    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=str, help='experiment id')

    # transfer learning
    parser.add_argument('-tf', '--transfer-learning', action='store_true', default=False,
                        help='turn on transfer learning')

    parser.add_argument('-tl','--tl-scheme', default='rtal', choices=['rtal',
                                                                'ftal'],
                        help='transfer learning scheme (default: rtal)')

    parser.add_argument('--type', default='none', help='fake key type, fake2-1, fake3_100S')
    parser.add_argument('--flipperc', default=0, type=float, help='flip percentange 0~1')

    args = parser.parse_args()

    if args.transfer_learning:
        args.lr = 0.001

    pprint(vars(args))

    exp = ClassificationPrivateExperiment(vars(args))

    if exp.is_tl:
        exp.transfer_learning()
    else:
        shutil.copytree('./passport_configs', str(exp.logdir) +"/passport_configs")
        shutil.copytree('./models/layers', str(exp.logdir) +"/models/layers")
        shutil.copy('train_v23.py', str(exp.logdir) +"/train_v23.py")
        exp.training()

    print('Training done at', exp.logdir)


if __name__ == '__main__':
    main()
