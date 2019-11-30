# -*- coding: utf-8 -*-

import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.cuda as cuda
import torchvision.transforms as transforms
import torchvision.utils as utl
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets
import torchvision.models as mdl

def checkEntropy(scores):
    scores = scores.squeeze()
    scr = scores.clone()
    scr[scr <= 0] = 1.0
    return - torch.sum(scores * torch.log(scr))
    

def checkConfidence(scores, K=10):
    scores = scores.squeeze()
    hScores, _ = torch.sort(scores, dim=0, descending=True)
    
    return hScores[0] / torch.sum(hScores[:K])
    

def integratedForward(model, sps, batchSize, nClasses, device='cpu', voteMethod='avg_softmax'):
    N = sps.size(0)
    feats = torch.empty(N, nClasses)
    model = model.to(device)
    
    with torch.no_grad():
        baseInx = 0
        while baseInx < N:
            cuda.empty_cache()
            endInx = min(baseInx + batchSize, N)
            y = model(sps[baseInx:endInx, :].to(device)).detach().to('cpu')
            feats[baseInx:endInx, :] = y
            baseInx = endInx
    
    if voteMethod == 'avg_feat':
        feat = torch.mean(feats, dim=0, keepdim=True)
    elif voteMethod == 'most_vote':
        maxV, _ = torch.max(feats, dim=1, keepdim=True)
        feat = torch.sum(feats == maxV, dim=0, keepdim=True)
    elif voteMethod == 'weighted_feat':
        feat = torch.mean(feats, dim=0, keepdim=True)
        maxV, _ = torch.max(feats, dim=1, keepdim=True)
        feat = feat * torch.sum(feats == maxV, dim=0, keepdim=True).float()
    elif voteMethod == 'avg_softmax':
        feats = nn.functional.softmax(feats, dim=1)
        feat = torch.mean(feats, dim=0, keepdim=True)
    else:
        # default method: avg_softmax
        feats = nn.functional.softmax(feats, dim=1)
        feat = torch.mean(feats, dim=0, keepdim=True)
    
    return feat, feats

# not updated, deprecated
def integratedForward_cls(model, sps, batchSize, nClasses, device='cpu', count_votes=False):
    N = sps.size(0)
    feats = torch.empty(N, nClasses)
    model = model.to(device)
    
    with torch.no_grad():
        baseInx = 0
        while baseInx < N:
            cuda.empty_cache()
            endInx = min(baseInx + batchSize, N)
            y = model.classifier(sps[baseInx:endInx, :].to(device)).detach().to('cpu')
            feats[baseInx:endInx, :] = y
            baseInx = endInx
    
    if count_votes:
        maxV, _ = torch.max(feats, dim=1, keepdim=True)
        feat = torch.sum(feats == maxV, dim=0, keepdim=True)
    else:
        feat = torch.mean(feats, dim=0, keepdim=True)
    
    return feat, feats


def findNeighbors_random(sp, K, r=[2], direction='both'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
        
    if isinstance(K, list):
        K = sum(K)
        
    # randomly select directions
    shifts = torch.randn(K, sp.size(1) * sp.size(2) * sp.size(3)).to('cuda')
    shifts = nn.functional.normalize(shifts, p=2, dim=1)
    shifts = shifts.view(K, sp.size(1), sp.size(2), sp.size(3)).contiguous()
    
    if direction == 'both':
        shifts = torch.cat([shifts, -shifts], dim=0)
    
    nbs = []
    for rInx in range(len(r)):
        nbs.append(sp.to('cuda') + r[rInx] * shifts)

    return torch.cat(nbs, dim=0)
    

def findNeighbors_plain_vgg(model, sp, K, r=[2], direction='both', device='cpu'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # place holder for input, and set to require gradient
    x = sp.clone().to(device)
    x.requires_grad = True
    
    # forward through the feature part
    y = model.features(x)
    y = model.avgpool(y)
    y = y.view(y.size(0), -1)
    
    # forward through classifier layer by layer
    for lyInx, module in model.classifier.named_children():
        # forward
        y = module(y)
        
        # at each layer activation
        if isinstance(module, nn.Linear):
            # for each neuron
            for i in range(y.size(1)):
                # clear previous gradients
                x.grad = None
                
                # compute gradients
                goal = torch.abs(y[0, i])
                goal.backward(retain_graph=True)    # retain graph for future computation
                
                # compute distance
                d = torch.abs(y[0, i]) / torch.norm(x.grad)
                
                # keep K shortest distances
                selected_list.append((d.clone().detach().to('cpu'), x.grad.clone().detach().to('cpu')))
                selected_list = sorted(selected_list, key=lambda x:x[0], reverse=False)
                selected_list = selected_list[0:K]
    
    # generate neighboring samples
    grad_list = [e[1] / torch.norm(e[1]) for e in selected_list]
    unit_shifts = torch.cat(grad_list, dim=0)
    nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs
    

def findNeighbors_lastLy_vgg(model, sp, K, r=[2], direction='both', device='cpu'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # place holder for input, and set to require gradient
    x = sp.clone().to(device)
    x.requires_grad = True
    
    # forward through the feature part
    y = model(x)
    y = y.view(y.size(0), -1)

    for i in range(y.size(1)):
        # clear previous gradients
        x.grad = None
                
        # compute gradients
        goal = torch.abs(y[0, i])
        if i < y.size(1) - 1:
            goal.backward(retain_graph=True)    # retain graph for future computation
        else:
            goal.backward(retain_graph=False)
                
        # compute distance
        d = torch.abs(y[0, i]) / torch.norm(x.grad)
                
        # keep K shortest distances
        selected_list.append((d.clone().detach().to('cpu'), x.grad.clone().detach().to('cpu')))
        selected_list = sorted(selected_list, key=lambda x:x[0], reverse=False)
        selected_list = selected_list[0:K]
    
    # generate neighboring samples
    grad_list = [e[1] / torch.norm(e[1]) for e in selected_list]
    unit_shifts = torch.cat(grad_list, dim=0)
    nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs
    
    
def findNeighbors_approx_vgg(model, sp, K, r=[2], direction='both', device='cpu'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # place holder for input, and set to require gradient
    x = sp.clone().to(device)
    x.requires_grad = True
    
    # forward through the feature part
    y = model.features(x)
    y = model.avgpool(y)
    y = y.view(y.size(0), -1)
    
    # forward through classifier layer by layer
    lnLy_inx = 0
    for lyInx, module in model.classifier.named_children():
        # forward
        y = module(y)
        
        # at each layer activation
        if isinstance(module, nn.Linear):
            KInx = min(lnLy_inx, len(K)-1)
            if K[KInx] > 0:
                with torch.no_grad():
                    # compute weight norm
                    w_norm = torch.norm(module.weight, dim=1, keepdim=True)
            
                    # compute distance
                    d = torch.abs(y) / w_norm.t()
                    _, sortedInx = torch.sort(d, dim=1, descending=False)
            
                # for each selected neuron
                for i in range(K[KInx]):
                
                    # clear previous gradients
                    x.grad = None
                
                    # compute gradients
                    goal = torch.abs(y[0, sortedInx[0, i]])
                    goal.backward(retain_graph=True)    # retain graph for future computation
                
                    # record gradients
                    selected_list.append(x.grad.clone().detach().to('cpu') / torch.norm(x.grad).detach().to('cpu'))
            
            # update number of linear layer sampled
            lnLy_inx = lnLy_inx + 1
    
    # generate neighboring samples
    unit_shifts = torch.cat(selected_list, dim=0)
    nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs
    
    
def findNeighbors_randPick_vgg(model, sp, K, r=[2], direction='both', device='cpu'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # place holder for input, and set to require gradient
    x = sp.clone().to(device)
    x.requires_grad = True
    
    # forward through the feature part
    y = model.features(x)
    y = model.avgpool(y)
    y = y.view(y.size(0), -1)
    
    # forward through classifier layer by layer
    lnLy_inx = 0
    for lyInx, module in model.classifier.named_children():
        # forward
        y = module(y)
        
        # at each layer activation
        if isinstance(module, nn.Linear):
            KInx = min(lnLy_inx, len(K)-1)
            if K[KInx] > 0:
                # randomly permute indices
                pickInx = torch.randperm(y.size(1))
            
                # for each selected neuron
                for i in range(K[KInx]):
                
                    # clear previous gradients
                    x.grad = None
                
                    # compute gradients
                    goal = torch.abs(y[0, pickInx[i]])
                    goal.backward(retain_graph=True)    # retain graph for future computation
                
                    # record gradients
                    selected_list.append(x.grad.clone().detach().to('cpu') / torch.norm(x.grad).detach().to('cpu'))
            
            # update number of linear layer sampled
            lnLy_inx = lnLy_inx + 1
    
    # generate neighboring samples
    unit_shifts = torch.cat(selected_list, dim=0)
    nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs

# not updated, deprecated
def findNeighbors_feats_lastLy_vgg(model, sp, K, r=[2], direction='both', device='cpu', includeOriginal=True):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # forward through the feature part
    with torch.no_grad():
        feat = model.features(sp.to(device))
        feat = feat.view(feat.size(0), -1).contiguous().detach()
        
    # place holder for feature, and set to require gradient
    x = feat.clone().detach()
    x.requires_grad = True
    
    # forward through the classifier part
    y = model.classifier(x)
    y = y.view(y.size(0), -1)

    for i in range(y.size(1)):
        # clear previous gradients
        x.grad = None
                
        # compute gradients
        goal = torch.abs(y[0, i])
        if i < y.size(1) - 1:
            goal.backward(retain_graph=True)    # retain graph for future computation
        else:
            goal.backward(retain_graph=False)
                
        # compute distance
        d = torch.abs(y[0, i]) / torch.norm(x.grad)
                
        # keep K shortest distances
        selected_list.append((d.clone().detach().to('cpu'), x.grad.clone().detach().to('cpu')))
        selected_list = sorted(selected_list, key=lambda x:x[0], reverse=False)
        selected_list = selected_list[0:K]
    
    # generate neighboring samples
    grad_list = [e[1] / torch.norm(e[1]) for e in selected_list]
    unit_shifts = torch.cat(grad_list, dim=0)
    if includeOriginal:
        nbs = [feat.to('cpu')]
    else:
        nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(feat.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(feat.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(feat.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(feat.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs

# not updated, deprecated
def findNeighbors_feats_approx_vgg(model, sp, K, r=[2], direction='both', device='cpu', includeOriginal=True):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # forward through the feature part
    with torch.no_grad():
        feat = model.features(sp.to(device))
        feat = feat.view(feat.size(0), -1).contiguous().detach()
        
    # place holder for feature, and set to require gradient
    x = feat.clone().detach()
    x.requires_grad = True
    y = x
    
    # forward through classifier layer by layer
    lnLy_inx = 0
    for lyInx, module in model.classifier.named_children():
        # forward
        y = module(y)
        
        # at each layer activation
        if isinstance(module, nn.Linear):
            KInx = min(lnLy_inx, len(K)-1)
            if K[KInx] > 0:
                with torch.no_grad():
                    # compute weight norm
                    w_norm = torch.norm(module.weight, dim=1, keepdim=True)
            
                    # compute distance
                    d = torch.abs(y) / w_norm.t()
                    _, sortedInx = torch.sort(d, dim=1, descending=False)
            
                # for each selected neuron
                for i in range(K[KInx]):
                
                    # clear previous gradients
                    x.grad = None
                
                    # compute gradients
                    goal = torch.abs(y[0, sortedInx[0, i]])
                    goal.backward(retain_graph=True)    # retain graph for future computation
                
                    # record gradients
                    selected_list.append(x.grad.clone().detach().to('cpu') / torch.norm(x.grad).detach().to('cpu'))
            
            # update number of linear layer sampled
            lnLy_inx = lnLy_inx + 1
    
    # generate neighboring samples
    unit_shifts = torch.cat(selected_list, dim=0)
    if includeOriginal:
        nbs = [feat.to('cpu')]
    else:
        nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(feat.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(feat.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(feat.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(feat.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs
    
    
def formSquad_vgg(method, model, sp, K, r=[2], direction='both', device='cpu', includeOriginal=True):
    if method == 'random':
        nbs = findNeighbors_random(sp, K, r, direction=direction)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'plain':
        nbs = findNeighbors_plain_vgg(model, sp, K, r, direction=direction, device=device)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'lastLy':
        nbs = findNeighbors_lastLy_vgg(model, sp, K, r, direction=direction, device=device)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'approx':
        nbs = findNeighbors_approx_vgg(model, sp, K, r, direction=direction, device=device)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'randPick':
        nbs = findNeighbors_randPick_vgg(model, sp, K, r, direction=direction, device=device)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'feats_lastLy':
        nbs = findNeighbors_feats_lastLy_vgg(model, sp, K, r, direction=direction, device=device, includeOriginal=includeOriginal)
    elif method == 'feats_approx':
        nbs = findNeighbors_feats_approx_vgg(model, sp, K, r, direction=direction, device=device, includeOriginal=includeOriginal)
    else:
        # if invalid method, use default setting. (actually should raise error here)
        nbs = findNeighbors_random(sp, K, r, direction=direction)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
        
    return nbs
    

def findNeighbors_approx_resnet(model, sp, K, r=[2], direction='both', device='cpu'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # place holder for input, and set to require gradient
    x = sp.clone().to(device)
    x.requires_grad = True
    
    # forward through the model
    y = model(x)
    y = y.view(y.size(0), -1)
    
    if K > 0:
        with torch.no_grad():
            # compute weight norm
            w_norm = torch.norm(model.fc.weight, dim=1, keepdim=True)
            
            # compute distance
            d = torch.abs(y) / w_norm.t()
            _, sortedInx = torch.sort(d, dim=1, descending=False)
            
        # for each selected neuron
        for i in range(K):
                
            # clear previous gradients
            x.grad = None
                
            # compute gradients
            goal = torch.abs(y[0, sortedInx[0, i]])
            goal.backward(retain_graph=True)    # retain graph for future computation
                
            # record gradients
            selected_list.append(x.grad.clone().detach().to('cpu') / torch.norm(x.grad).detach().to('cpu'))
    
    # generate neighboring samples
    unit_shifts = torch.cat(selected_list, dim=0)
    nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs
    

def findNeighbors_approx_resnet_small(model, sp, K, r=[2], direction='both', device='cpu'):
    # only accept single sample
    if sp.size(0) != 1:
        return None
    
    # storages for K selected distances and linear mapping
    selected_list = []
    
    # set model to evaluation mode
    model = model.to(device)
    model = model.eval()
    
    # place holder for input, and set to require gradient
    x = sp.clone().to(device)
    x.requires_grad = True
    
    # forward through the model
    y = model(x)
    y = y.view(y.size(0), -1)
    
    if K > 0:
        with torch.no_grad():
            # compute weight norm
            w_norm = torch.norm(model.linear.weight, dim=1, keepdim=True)
            
            # compute distance
            d = torch.abs(y) / w_norm.t()
            _, sortedInx = torch.sort(d, dim=1, descending=False)
            
        # for each selected neuron
        for i in range(K):
                
            # clear previous gradients
            x.grad = None
                
            # compute gradients
            goal = torch.abs(y[0, sortedInx[0, i]])
            goal.backward(retain_graph=True)    # retain graph for future computation
                
            # record gradients
            selected_list.append(x.grad.clone().detach().to('cpu') / torch.norm(x.grad).detach().to('cpu'))
    
    # generate neighboring samples
    unit_shifts = torch.cat(selected_list, dim=0)
    nbs = []
    for rInx in range(len(r)):
        if direction == 'inc':
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
        elif direction == 'dec':
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
        else:
            nbs.append(sp.to('cpu') + r[rInx] * unit_shifts)
            nbs.append(sp.to('cpu') - r[rInx] * unit_shifts)
    nbs = torch.cat(nbs, dim=0)
    nbs = nbs.detach()
    nbs.requires_grad = False
    
    return nbs
    
    
def formSquad_resnet(method, model, sp, K, r=[2], direction='both', device='cpu', includeOriginal=True):
    if method == 'random':
        nbs = findNeighbors_random(sp, K, r, direction=direction)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'approx':
        nbs = findNeighbors_approx_resnet(model, sp, K, r, direction=direction, device=device)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    elif method == 'approx_cifar10':
        nbs = findNeighbors_approx_resnet_small(model, sp, K, r, direction=direction, device=device)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
    else:
        # if invalid method, use default setting. (actually should raise error here)
        nbs = findNeighbors_random(sp, K, r, direction=direction)
        if includeOriginal:
            nbs = torch.cat([sp, nbs], dim=0)
        
    return nbs
