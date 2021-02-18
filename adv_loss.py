import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def adv_loss_pgd(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
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
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.cross_entropy(model(x_adv), y)
    '''loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))'''
    loss = (loss_natural + beta * loss_robust)/(1.0 + beta)
    return loss


#==================== L2L_0 Loss ============================
def adv_loss_l2l_0(model,
                attacker,
                x_natural,
                y,
                optimizer,
                optimizer_att,
                beta=1.0,
                for_attacker = 0):


    if for_attacker == 0:
        model.train()
        attacker.eval()
    else:
        model.eval()
        attacker.train()
        
    batch_size = len(x_natural)
    advinput = x_natural
    
    # generate adversarial example
    perturbation = attacker(advinput)
    x_adv = x_natural + perturbation


    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    optimizer.zero_grad()
    optimizer_att.zero_grad()

    loss_robust = F.cross_entropy(model(x_adv), y)
    loss = loss_robust
    return loss


#=================== L2L_1 Loss ===========================
def adv_loss_l2l_1(model,
                attacker,
                x_natural,
                y,
                optimizer,
                optimizer_att,
                beta=1.0,
                epsilon = 0.031,
                for_attacker = 0):


    if for_attacker == 0:
        model.train()
        attacker.eval()
    else:
        model.eval()
        attacker.train()

    batch_size = len(x_natural)
    x_natural.requires_grad_()
    with torch.enable_grad():
        loss_natural = F.cross_entropy(model(x_natural), y)
    grad = torch.autograd.grad(loss_natural, [x_natural])[0]
    advinput = torch.cat([x_natural,1.0*(grad/grad.abs().max())], 1).detach()
    
    # generate adversarial example
    perturbation = attacker(advinput)
    x_adv = x_natural + perturbation


    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    optimizer.zero_grad()
    optimizer_att.zero_grad()

    loss_robust = F.cross_entropy(model(x_adv), y)

    loss = loss_robust
    return loss



#=================== L2L_2 Loss =============================
def adv_loss_l2l_2(model,
                attacker,
                x_natural,
                y,
                optimizer,
                optimizer_att,
                epsilon=0.031, 
                beta=1.0,
                for_attacker = 0):


    if for_attacker == 0:
        model.train()
        attacker.eval()
    else:
        model.eval()
        attacker.train()
        
    batch_size = len(x_natural)
    x_natural.requires_grad_()
    with torch.enable_grad():
        loss_natural = F.cross_entropy(model(x_natural), y)
    grad = torch.autograd.grad(loss_natural, [x_natural])[0]
    advinput = torch.cat([x_natural,1.0*(grad/grad.abs().max())], 1).detach()
    
    # generate adversarial example
    perturbation = attacker(advinput)
    x_adv = x_natural + perturbation
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        loss_adv = F.cross_entropy(model(x_adv), y)
    grad_adv = torch.autograd.grad(loss_adv, [x_adv])[0]
    advinput_1 = torch.cat([x_adv,1.0*(grad_adv/grad_adv.abs().max())], 1)
    perturbation_1 = attacker(advinput_1)

    perturbation_total = perturbation + perturbation_1
    perturbation_total = torch.clamp(perturbation_total, -epsilon, epsilon)

    x_adv_final = x_natural + perturbation_total
    x_adv_final = torch.clamp(x_adv_final, 0.0, 1.0)

    
    optimizer.zero_grad()
    optimizer_att.zero_grad()

    loss_robust = F.cross_entropy(model(x_adv_final), y)

    loss = loss_robust
    return loss



#=================== L2L_k Loss =============================
def adv_loss_l2l_k(k,
                model,
                attacker,
                x_natural,
                y,
                optimizer,
                optimizer_att,
                step_size = 0.007,
                epsilon=0.031, 
                beta=1.0,
                for_attacker = 0):


    if for_attacker == 0:
        model.train()
        attacker.eval()
    else:
        model.eval()
        attacker.train()
        
    batch_size = len(x_natural)

    x_adv = Variable(x_natural.data, requires_grad = True)
    for _ in range(k):
        with torch.enable_grad():
            loss_adv = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_adv, [x_adv])[0]
        advinput = torch.cat([x_adv,1.0*(grad/grad.abs().max())], 1).detach()
        
        # generate adversarial example
        perturbation = attacker(advinput)
        x_adv = x_adv + perturbation*step_size/epsilon
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    perturbation_total = x_adv - x_natural
    perturbation_total = torch.clamp(perturbation_total, -epsilon, epsilon)

    x_adv_final = x_natural + perturbation_total
    x_adv_final = torch.clamp(x_adv_final, 0.0, 1.0)


    optimizer.zero_grad()
    optimizer_att.zero_grad()
    
    # calculate robust loss
    loss_robust = F.cross_entropy(model(x_adv_final), y)

    loss = loss_robust
    return loss

