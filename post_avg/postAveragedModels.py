# -*- coding: utf-8 -*-

import robustml
import numpy as np
from collections import OrderedDict
import post_avg.PADefense as padef
import post_avg.resnetSmall as rnsmall

import torch
import torchvision.models as mdl
import torchvision.transforms as transforms

class PostAveragedResNet152(robustml.model.Model):
    def __init__(self, K, R, eps, device='cuda'):
        self._model = mdl.resnet152(pretrained=True).to(device)
        self._dataset = robustml.dataset.ImageNet((224, 224, 3))
        self._threat_model = robustml.threat_model.Linf(epsilon=eps)
        self._K = K
        self._r = [R/3, 2*R/3, R]
        self._sample_method = 'random'
        self._vote_method = 'avg_softmax'
        self._device = device
    
    @property
    def model(self):
        return self._model
    
    @property
    def dataset(self):
        return self._dataset
        
    @property
    def threat_model(self):
        return self._threat_model
        
    def classify(self, x):
        x = x.unsqueeze(0)
        
        # gather neighbor samples
        x_squad = padef.formSquad_resnet(self._sample_method, self._model, x, self._K, self._r, device=self._device)
        
        # forward with a batch of neighbors
        logits, _ = padef.integratedForward(self._model, x_squad, batchSize=100, nClasses=1000, device=self._device, voteMethod=self._vote_method)

        return torch.as_tensor(logits)

    def __call__(self, x):
        logits_list = []
        for img in x:
            logits = self.classify(img)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0)
        
    def _preprocess(self, image):
        # normalization used by pre-trained model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return normalize(image)
        
    def to(self, device):
        self._model = self._model.to(device)
        self._device = device
        
    def eval(self):
        self._model = self._model.eval()
        
        
def pa_resnet152_config1():
    return PostAveragedResNet152(K=15, R=30, eps=8/255)

    
class PostAveragedResNet110(robustml.model.Model):
    def __init__(self, K, R, eps, device='cuda'):
        # load model state dict
        checkpoint = torch.load('post_avg/trainedModel/resnet110.th')
        paramDict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            # remove 'module.' prefix introduced by DataParallel, if any
            if k.startswith('module.'):
                paramDict[k[7:]] = v
        self._model = rnsmall.resnet110()
        self._model.load_state_dict(paramDict)
        self._model = self._model.to(device)
        
        self._dataset = robustml.dataset.CIFAR10()
        self._threat_model = robustml.threat_model.Linf(epsilon=eps)
        self._K = K
        self._r = [R/3, 2*R/3, R]
        self._sample_method = 'random'
        self._vote_method = 'avg_softmax'
        self._device = device
    
    @property
    def model(self):
        return self._model
    
    @property
    def dataset(self):
        return self._dataset
        
    @property
    def threat_model(self):
        return self._threat_model
        
    def classify(self, x):
        x = x.unsqueeze(0)

        # gather neighbor samples
        x_squad = padef.formSquad_resnet(self._sample_method, self._model, x, self._K, self._r, device=self._device)
        
        # forward with a batch of neighbors
        logits, _ = padef.integratedForward(self._model, x_squad, batchSize=1000, nClasses=10, device=self._device, voteMethod=self._vote_method)
        
        return torch.as_tensor(logits)

    def __call__(self, x):
        logits_list = []
        for img in x:
            logits = self.classify(img)
            logits_list.append(logits)
        return torch.cat(logits_list, dim=0)

    def _preprocess(self, image):
        # normalization used by pre-trained model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return normalize(image)
        
    def to(self, device):
        self._model = self._model.to(device)
        self._device = device
        
    def eval(self):
        self._model = self._model.eval()
        
        
def pa_resnet110_config1():
    return PostAveragedResNet110(K=15, R=6, eps=8/255)
