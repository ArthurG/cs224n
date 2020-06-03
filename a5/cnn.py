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
    def __init__(self, m_word_size, e_char_size, num_filters, k=5, padding=1):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char_size, out_channels=num_filters, kernel_size=k, padding=padding, bias=True)
        pool_size = m_word_size + 2 * padding - k + 1
        self.maxpool = nn.MaxPool1d(pool_size)

    def forward(self, x_reshaped):
        x_conv = self.conv(x_reshaped)
        x_conv_out = self.maxpool(F.relu(x_conv)).squeeze(2)
        return x_conv_out

    ### END YOUR CODE

