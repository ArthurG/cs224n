#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, e_word_size):
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(in_features = e_word_size, out_features = e_word_size, bias=True)
        self.w_gate = nn.Linear(in_features = e_word_size, out_features = e_word_size, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out):
        # In: (batch_size, e_word_size)
        # Out: (batch_size, e_word_size)

        x_proj = F.relu(self.w_proj(x_conv_out))
        x_gate = self.sigmoid(self.w_proj(x_conv_out))
        x_highway = x_gate * x_proj + (1- x_gate) * x_conv_out
        return x_highway

    ### END YOUR CODE

