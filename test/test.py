#!/usr/bin/env python
# coding=utf-8

#########################################################################
#
# Copyright (c) 2018 ICAIS Lab. All Rights Reserved
#
#########################################################################


"""
File: test.py
Author: fxsun@ICAIS Lab
Date: 2018/12/26 15:12:59
Brief: 
"""

import torch.nn as nn
import copy

class a(nn.Module):
    def __init__(self):
        super(a, self).__init__()
        self.conv1 = nn.Conv2d(1,1,1)

aa = a()
print(aa.conv1)
