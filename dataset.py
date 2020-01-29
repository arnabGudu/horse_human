#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 02:46:16 2020

@author: lucifer
"""


import torch
import torchvision
import torchvision.transforms as tm

def dataloader(filename = 'data'):
    transfrom = tm.Compose(
        [tm.ToTensor(),
         tm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.ImageFolder(root = filename + '/train', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.ImageFolder(root = filename + '/validation', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    
    return trainloader, testloader, trainset.classes
