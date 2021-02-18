'''
Utility functions, PGD attacks and Loss functions
'''
import math
import numpy as np
import random
import scipy.io
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable


device = 'cuda' if torch.cuda.is_available() else 'cpu'





class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, target), 1)

        return loss



def cos_dist(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    batch_size = x.size(0)
    c = torch.clamp(1 - cos(x.view(batch_size, -1), y.view(batch_size, -1)),
                    min=0)
    return c.mean()


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


