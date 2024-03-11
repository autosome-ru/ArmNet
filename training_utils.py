import math
import random
from pathlib import Path
import os, gc
from typing import Literal

from fastai.vision.all import DataLoaders, Metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, RandomSampler, DataLoader, Dataset

import numpy as np 


def parameter_count(model):
    pars = 0
    for _, p  in model.named_parameters():
        pars += torch.prod(torch.tensor(p.shape))
    return pars.item()

def loss(pred,target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore

class MAE(Metric):
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.x,self.y = [],[]
        
    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    
class MAE_2A3(MAE):
    def __init__(self): 
        super().__init__()
        
    def accumulate(self, learn):
        x = learn.pred[:, :, 0][learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][:, :, 0][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)

    
class MAE_DMS(MAE):
    def __init__(self): 
        super().__init__()
        
    def accumulate(self, learn):
        x = learn.pred[:, :, 1][learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][:, :, 1][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)   
        
        
def val_to(x, device="cuda"):
    if isinstance(x, list):
        return [val_to(z) for z in x]
    return x.to(device)

def dict_to(x, device='cuda'):
    return {k: val_to(x[k], device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
            
def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    batch_count: int,
    num_workers: int,
    device: torch.device,
    no_weights: bool = False,   
) -> DataLoaders:
    
    if no_weights:
        sampler_train = RandomSampler(
            train_dataset,
            replacement = True,
            num_samples = batch_size * batch_count
        )
    else:
        sampler_train = WeightedRandomSampler(
            weights = train_dataset.weights, 
            num_samples = batch_size * batch_count)


    train_dataloader = DeviceDataLoader(
        DataLoader(
            dataset = train_dataset, 
            batch_size = batch_size,
            sampler = sampler_train,
            num_workers = num_workers,
            persistent_workers=True),
        device = device)


    val_dataloader = DeviceDataLoader(
        DataLoader(
            dataset = val_dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            persistent_workers = True,
            shuffle=False),
        device = device)

    return DataLoaders(train_dataloader, val_dataloader)
            
