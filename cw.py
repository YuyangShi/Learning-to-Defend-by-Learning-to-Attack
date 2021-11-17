import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

def norm_fn_wapper(type):
    return lambda x : x.view([x.shape[0],-1]).norm(dim=1, p=float(type)).view(-1,1,1,1)

class Model_Wapper:
    def __init__(self, model, loss_fn = nn.CrossEntropyLoss()):
        self._model = model
        self.loss_fn = loss_fn

    def predictions_gradient(self, images, label):
        images = images.clone()
        images.requires_grad_()

        predictions = self._model(images)

        loss = self.loss_fn(predictions, label)
        loss.backward()
        grad = images.grad
        self._model.zero_grad()

        return predictions, grad

    def gradient(self, image, label):
        p, g = self.predictions_gradient(image, label)
        return g
    
class CWAttack:
    def __init__(self, model = None, eps = 1, bounds = (0.,1.), type = "inf", torch_trans = (0.,1.),
                learning_rate = 5e-3, max_iterations = 50, abort_early = True,
                initial_const = 1e-5, largest_const = 1e+2, reduce_const = False,
                decrease_factor = 0.9, const_factor = 10.0, verbose = False):
        """
        The L_infinity optimized attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        reduce_const: If true, after each successful attack, make const smaller.
        decrease_factor: Rate at which we should decrease tau, less than one.
          Larger produces better quality results.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        """
        assert type == "inf"
        self.verbose = verbose
        self._model = model
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor



        self.eps = eps
        self._min = bounds[0]
        self._max = bounds[1]
        self.std = torch_trans[1]
        self.mean = torch_trans[0]
        self.norm_fn = norm_fn_wapper(type)

    def __call__(self, image, label):
        perturbed = image.clone()
        for i in range(len(image)):
            if self._model(image[i][None,:,:,:]).argmax(dim=1) == label[i]:
                perturbed[i] = self.attack_single(image[i][None,:,:,:], label[i])
        return perturbed

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the previous image
        prev = img.clone()
        tau = 1.0
        const = self.INITIAL_CONST

        while tau > 1./256:
            # try to solve given this tau value
            res = self.gradient_descent(img.detach(), target, prev.detach(), tau, const)
            if res == None:
                # the attack failed, we return this as our final answer
                if self.verbose:
                    print("Fail")
                return img

            scores, origscores, nimg, const = res
            if self.REDUCE_CONST: const /= 2

            # the attack succeeded, reduce tau and try again
            actualtau = ((nimg-img)*self.std).abs().max()
            if self.verbose:
                print("actualtau: ",actualtau.item())
            if actualtau < self.eps:
                if self.verbose:
                    print("Success")
                return nimg

            if actualtau < tau:
                tau = actualtau
            if self.verbose:
                print("Tau",tau)

            prev = nimg
            tau *= self.DECREASE_FACTOR
        if self.verbose:
            print("Fail")
        return img

    def gradient_descent(self, oimgs, labs, starts, tt, CONST):
        # oimgs = oimgs*self.std + self.mean
        # oimgs = oimgs*2-1
        starts = starts*self.std + self.mean
        starts = starts*2-1
        def torchacrtanh(x):
            x = x*0.9999
            return 0.5* ((1+x)/(1-x)).log()
        # imgs = torchacrtanh(oimgs)
        starts = torchacrtanh(starts)
        orig_output = self._model(oimgs)

        modifier = torch.zeros(oimgs.shape).to(oimgs.device)
        modifier.requires_grad_()
        optimizer = torch.optim.Adam([modifier], lr = self.LEARNING_RATE)
        #optimizer = torch.optim.SGD([modifier], lr = self.LEARNING_RATE, momentum=0.5)
        while CONST < self.LARGEST_CONST:
            # try solving for each value of the constant
            if self.verbose:
                print('try const', CONST)
            for step in range(self.MAX_ITERATIONS):
                # feed_dict={timg: imgs,
                #            tlab:labs,
                #            tau: tt,
                #            simg: starts,
                #            const: CONST}
                # if step%(self.MAX_ITERATIONS//10) == 0:
                #     print(loss,loss1,loss2)

                # perform the update step

                self._model.zero_grad()
                optimizer.zero_grad()

                newimg = (modifier + starts.detach()).tanh()/2+0.5
                output = self._model((newimg-self.mean)/self.std)

                real = output[:,labs].sum()
                output2 = output.clone()
                output2[:,labs] -= 10000
                other = output2.max()
                loss1 = F.relu(real-other)
                #print(((oimgs*self.std + self.mean)).max().item())
                loss2 = (F.relu((newimg-oimgs*self.std - self.mean).abs()-tt)).sum()
                #print(loss1.item(), loss2.item())
                loss = CONST*loss1 + loss2

                loss.backward()
                optimizer.step()
                #if self.verbose:
                #    print(loss.item(),loss1.item(),loss2.item())
                # it worked
                if output.argmax()!=labs and (newimg-oimgs*self.std - self.mean).abs().max() < self.eps:
                    #print("HITTTT", (newimg-oimgs*self.std - self.mean).abs().max() )
                    return output, orig_output, ((newimg-self.mean)/self.std).detach(), CONST
                if loss.item() < .0001*CONST and self.ABORT_EARLY:
                    if output.argmax()!=labs:
                        return output, orig_output, ((newimg-self.mean)/self.std).detach(), CONST

            # we didn't succeed, increase constant and try again
            CONST *= self.const_factor
