import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps,min_v,max_v):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), min_v,max_v)
    # Return perturbed samples
    return x_adv

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl,rot):
    dat = data.clone().detach()
    dat.requires_grad = True
    if rot:
        out,_ = model(dat)
    else:
        out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start,min_v,max_v,rot):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    x = x_nat + torch.randn(x_nat.shape).to(device).uniform_(-eps, eps) if rand_start else x_nat.clone()
    # Make sure the sample is projected into original distribution bounds [0,1]
    x = torch.clamp(x, min_v,max_v)
    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        gradient = gradient_wrt_data(model,device,x,lbl,rot)
        # Perturb the image using the gradient
        x += torch.sign(gradient)*alpha
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        x = torch.clamp(x-x_nat,-eps,eps)
        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        x = torch.clamp(x+x_nat,min_v,max_v)
    # Return the final perturbed samples
    return x


def FGSM_attack(model, device, dat, lbl, eps,min_v,max_v,rot):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: FGSM is a special case of PGD
    return PGD_attack(model, device, dat, lbl, eps, eps, 1, False,min_v,max_v,rot)



def rFGSM_attack(model, device, dat, lbl, eps,min_v,max_v,rot):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: rFGSM is a special case of PGD
    return PGD_attack(model, device, dat, lbl, eps, eps, 1, True,min_v,max_v,rot)


def FGM_L2_attack(model, device, dat, lbl, eps,min_v,max_v,rot):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()
    x = x_nat.clone()
    # Compute gradient w.r.t. data
    gradient = gradient_wrt_data(model,device,x,lbl,rot)
    # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
    # HINT: Flatten gradient tensor first, then compute L2 norm
    l2_of_grad = torch.norm(gradient.view(gradient.shape[0],-1),dim=1)
    # Perturb the data using the gradient
    # HINT: Before normalizing the gradient by its L2 norm, use
    # torch.clamp(l2_of_grad, min=1e-12) to prevent division by 0
    l2_of_grad = torch.clamp(l2_of_grad, min=1e-12)
    # Add perturbation the data
    x = x_nat + gradient*eps/l2_of_grad.view(l2_of_grad.shape[0],1,1,1)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x = torch.clamp(x, min_v,max_v)
    # Return the perturbed samples
    return x
