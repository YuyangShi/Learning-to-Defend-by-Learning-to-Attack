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

from models.resnet import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type = float,
                    help='perturbation')
parser.add_argument('--num-steps', default=[10,20,100], type = int, nargs = '+',
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, type = float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--which', default = 'best', type = str)
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
parser.add_argument('--model_file', default='models.wideresnet', type = str,
                    help = 'model')
parser.add_argument('--gamma', default=1.0, type = float)


args = parser.parse_args()

exec('from '+args.model_file + ' import *')

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    
    model.eval()
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        #X_pgd = Variable(X.detach() + 0.001 * torch.randn(X.shape).cuda().detach(), requires_grad= True)
    for _ in range(num_steps):
        X_pgd.requires_grad_()
        #X_pgd.zero_grad()
        with torch.enable_grad():
            loss = F.cross_entropy(args.gamma * model(X_pgd), y)
        grad = torch.autograd.grad(loss, [X_pgd])[0]
        X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
        X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
        X_pgd = torch.clamp(X_pgd, 0.0, 1.0)
        
        '''loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)'''

        '''X_pgd = Variable(X_pgd.data + step_size * torch.sign(grad.detach()))
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
        #X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0.0, 1.0), requires_grad=True)'''
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()

    for i in args.num_steps:
        robust_err_total = 0
        natural_err_total = 0
        total = 0.0
        print('-------------------------------- PGD '+str(i) + ' White Box ----------------------------')
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_whitebox(model, X, y, num_steps = i)
            robust_err_total += err_robust
            natural_err_total += err_natural
            total += args.test_batch_size
        print('natural_err_total: ', natural_err_total,  'Acu:', 1.0 - natural_err_total.detach().cpu().numpy()/total)
        print('robust_err_total: ', robust_err_total, 'Acu:', 1.0 - robust_err_total.detach().cpu().numpy()/total)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()

    for i in args.num_steps:
        robust_err_total = 0
        natural_err_total = 0
        total = 0.0
        print('-------------------------------- PGD '+str(i) + ' Black Box ----------------------------')

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y, num_steps = i)
            robust_err_total += err_robust
            natural_err_total += err_natural
            total += args.test_batch_size
        print('natural_err_total: ', natural_err_total, 'Acu:', 1.0 - natural_err_total.detach().cpu().numpy()/total)
        print('robust_err_total: ', robust_err_total, 'Acu:',1.0 - robust_err_total.detach().cpu().numpy()/total)


def main():

    if args.white_box_attack:
        # white-box attack
        print('-----------------------------------------------------------')
        print('pgd white-box attack')
        print('num_steps:', args.num_steps, 'step_size:', args.step_size)
        print('Model:', args.model)

        model = eval(args.model+'()').to(device)
        if args.which == 'last':
            model.load_state_dict(torch.load(args.model_path + '/checkpoint.pth.tar')['state_dict'])
        else:
            model.load_state_dict(torch.load(args.model_path + '/model_best.pth.tar')['state_dict'])
        '''try:
            model.load_state_dict(torch.load(args.model_path))
        except:
            model.load_state_dict(torch.load(args.model_path + '/model_best.pth.tar')['state_dict'])'''

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print('----------------------------------------------------------')
        print('pgd black-box attack')
        print('num_steps:', args.num_steps, 'step_size:', args.step_size)
        print('Model:', args.model)
        model_target = eval(args.model+'()').to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = eval(args.model+'()').to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
