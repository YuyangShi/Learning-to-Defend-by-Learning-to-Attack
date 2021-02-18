from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from cw import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type = float,
                    help='perturbation')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path', type = str,
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path', type = str,
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path', type = str,
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--model', default='WideResNet', type = str,
                    help = 'model')
parser.add_argument('--model-target', default='WideResNet', type = str,
                    help = 'target model')
parser.add_argument('--model-source', default='WideResNet', type = str,
                    help = 'source model')
parser.add_argument('--max_batch', default=None, type = int,
                    help = 'source model')
args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon):
    attacker = CWAttack(model = model, eps = epsilon)
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_cw = attacker(X, y)
    err_cw = (model(X_cw).data.max(1)[1] != y.data).float().sum()
    print('err cw (white-box): ', err_cw)
    return err, err_cw




def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()

    robust_err_total = 0
    natural_err_total = 0
    total = 0.0
    print('-------------------------------- CW White Box ----------------------------')
    n = 0
    flag = args.max_batch is not None
    for data, target in test_loader:
        if flag and n > args.max_batch:
            break
        else:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data), Variable(target)
            err_natural, err_robust = _cw_whitebox(model, X, y)
            robust_err_total += err_robust
            natural_err_total += err_natural
            total += args.test_batch_size
            n += 1
    print('natural_err_total: ', natural_err_total,  'Acu:', 1.0 - natural_err_total.detach().cpu().numpy()/total)
    print('robust_err_total: ', robust_err_total, 'Acu:', 1.0 - robust_err_total.detach().cpu().numpy()/total)




def main():
    # white-box attack
    print('-----------------------------------------------------------')
    print('cw white-box attack')
    print('Max Number of Batchs:', args.max_batch)
    print('Model:', args.model)
    model = eval(args.model+'(num_classes = 100)').to(device)
    try:
        model.load_state_dict(torch.load(args.model_path))
    except:
        model.load_state_dict(torch.load(args.model_path + '/checkpoint.pth.tar')['state_dict'])

    eval_adv_test_whitebox(model, device, test_loader)


if __name__ == '__main__':
    main()
