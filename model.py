#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:55:08 2020

@author: lucifer
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fn

class Model(nn.Module):
    def __init__(self, in_feature = 3, out_feature = 2, img_height = 300, img_width = 300):
        super(model, self).__init__()
        
        self.pad = nn.ZeroPad(2)
        self.conv1 = nn.Conv2d(in_feature, 6, 5)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * img_height * img_width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.linear(64, out_feature)
        
    def forward(self, img):
        hidden = self.pool(fn.relu(self.conv1(self.pad(img))))
        hidden = self.pool(fn.relu(self.conv2(self.pad(hidden))))
        hidden = hidden.view(-1, 16 * img.shape[0] * img.shape[1])
        hidden = fn.relu(self.fc1(hidden))
        hidden = fn.relu(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden
    