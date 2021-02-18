from __future__ import print_function
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import shutil

from models.wideresnet import *
from models.resnet import *
from attacker import *
from adv_loss import adv_loss_l2l_1

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                    metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_att', type=float, default=0.001,
                    help='attacker learning rate')
parser.add_argument('--opt_att', type = str, default = 'Adam', \
                    help = 'attacker optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type = float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type = int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type = float, 
                    help='perturb step size')
parser.add_argument('--beta', default=1.0, type = float, 
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet', type = str,
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model', default = 'WideResNet', type = str, \
                    help = 'classifier model')
parser.add_argument('--att_iter', default = 1, type = int, \
                    help = 'number of iterations for attacker update')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def save_checkpoint(state, is_best, epoch, savepath):
    torch.save(state, savepath+'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(savepath+'/checkpoint.pth.tar', savepath+'/model_best.pth.tar')
        
def train(args, model, attacker, device, train_loader, optimizer, optimizer_att, epoch):
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        model.zero_grad()
        attacker.zero_grad()
        # calculate robust loss
        loss = adv_loss_l2l_1(model=model,
                              attacker = attacker,
                              x_natural=data,
                              y=target,
                              optimizer=optimizer,
                              optimizer_att = optimizer_att,
                              beta=args.beta, for_attacker = 0)
        loss.backward()
        optimizer.step()

        for _ in range(args.att_iter):
            optimizer_att.zero_grad()

            attacker.zero_grad()
            model.zero_grad()
            loss_adv = -adv_loss_l2l_1(model=model,
                                  attacker = attacker,
                                  x_natural=data,
                                  y=target,
                                  optimizer=optimizer,
                                  optimizer_att = optimizer_att, 
                                  beta=args.beta, for_attacker = 1)
            loss_adv.backward()
            optimizer_att.step()


        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_test_adv(model, device, test_loader, \
                  step_size = 0.003, \
                  epsilon = 0.031, \
                  perturb_steps = 10,\
                  beta = 1.0,\
                  distance = 'l_inf'):
    model.eval()
    test_loss = 0
    correct = 0
    
    for x_natural, y in test_loader:
        x_natural, y = x_natural.to(device), y.to(device)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        # Generate Adversary
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_adv = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_adv, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_adv = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_adv, [x_adv])[0]
                for idx_batch in range(batch_size):
                    grad_idx = grad[idx_batch]
                    grad_idx_norm = l2_norm(grad_idx)
                    grad_idx /= (grad_idx_norm + 1e-8)
                    x_adv[idx_batch] = x_adv[idx_batch].detach() + step_size * grad_idx
                    eta_x_adv = x_adv[idx_batch] - x_natural[idx_batch]
                    norm_eta = l2_norm(eta_x_adv)
                    if norm_eta > epsilon:
                        eta_x_adv = eta_x_adv * epsilon / l2_norm(eta_x_adv)
                    x_adv[idx_batch] = x_natural[idx_batch] + eta_x_adv
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        
        output_adv = model(x_adv)
        test_loss += F.cross_entropy(output_adv, y, size_average=False).item()
        pred = output_adv.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('Test Adv: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 30:
        lr = args.lr * 0.1
    if epoch >= 60:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    model = eval(args.model + '()').to(device)
    attacker = WideAttacker(eps = args.epsilon, input_channel = 6).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.opt_att == 'SGD':
        optimizer_att = optim.SGD(attacker.parameters(), lr=args.lr_att, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer_att = eval('optim.'+args.opt_att)(attacker.parameters(), lr=args.lr_att, weight_decay=args.weight_decay)
    best_prec1 = 0
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        start = time.perf_counter()
        train(args, model, attacker, device, train_loader, optimizer, optimizer_att, epoch)
        elapsed = (time.perf_counter() - start)
        print('Time Elapsed:', elapsed)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        _,acu_top1 = eval_test(model, device, test_loader)
        _,adv_acu_top1 = eval_test_adv(model, device, test_loader, \
                  step_size = 0.003, \
                  epsilon = 0.031, \
                  perturb_steps = 10,\
                  beta = 1.0,\
                  distance = 'l_inf')
        print('================================================================')

        # save checkpoint
        is_best = adv_acu_top1 >= best_prec1
        best_prec1 = max(adv_acu_top1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': acu_top1,
            'attacker_state_dict': attacker.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch, savepath = model_dir)


if __name__ == '__main__':
    main()
