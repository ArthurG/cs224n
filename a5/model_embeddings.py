#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        # print("word embed size", word_embed_size)

        self.word_embed_size = word_embed_size
        self.char_embed_size = 50

        self.source = nn.Embedding(len(vocab.char2id), self.char_embed_size, vocab.char2id['∏'])

        self.cnn = CNN(self.char_embed_size, word_embed_size)
        self.highway = Highway(word_embed_size, 0.3)

        ### YOUR CODE HERE for part 1h

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        # print("input shape ", input.shape) # 10, 5, 21
        x_emb = self.source(input)
        # print("x_emb shape ", x_emb.shape) # 10, 5, 21, 50
        #x_emb = x_emb.transpose(2, 3)
        #print(x_emb)
        x_conv_out = torch.cat([self.cnn(row.squeeze(dim=0)).unsqueeze(dim=0) for row in x_emb.split(1)])
        #print(x_conv_out)
        # print("x_conv shape ", x_conv_out.shape)
        x_hwy = self.highway(x_conv_out)

        return x_hwy


        ### END YOUR CODE

