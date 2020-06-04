#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char_size, num_filters, k=5, padding=1):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char_size, out_channels=num_filters, kernel_size=k, padding=padding, bias=True)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x_reshaped):
        x_conv = self.conv(x_reshaped.transpose(1, 2))
        x_conv_out = self.maxpool(F.relu(x_conv)).squeeze(dim=2)
        return x_conv_out

    ### END YOUR CODE

