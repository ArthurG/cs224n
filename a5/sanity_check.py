#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1e
    sanity_check.py 1f
    sanity_check.py 1g
    sanity_check.py 1h
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, batch_iter, read_corpus
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT
from highway import Highway
from cnn import CNN


import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0

class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['<pad>']
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1e_sanity_check():
    """ Sanity check for to_input_tensor_char() function.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1e: To Input Tensor Char")
    print ("-"*80)
    vocabEntry = VocabEntry()

    print("Running test on a list of sentences")
    sentences = [['Human', ':', 'What', 'do', 'we', 'want', '?'], ['Computer', ':', 'Natural', 'language', 'processing', '!'], ['Human', ':', 'When', 'do', 'we', 'want', 'it', '?'], ['Computer', ':', 'When', 'do', 'we', 'want', 'what', '?']]
    sentence_length = 8
    BATCH_SIZE = 4
    word_length = 12
    output = vocabEntry.to_input_tensor_char(sentences, 'cpu')
    output_expected_size = [sentence_length, BATCH_SIZE, word_length]
    assert list(output.size()) == output_expected_size, "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))

    print("Sanity Check Passed for Question 1e: To Input Tensor Char!")
    print("-"*80)

def question_1f_sanity_check():
    """ Sanity check for Highway() class.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1f: Highway Class")


    """ Shape Dimension Check for Highway Class """
    inputs =torch.randint(100,size=(5, 100), dtype=torch.float)
    highway = Highway(100, 0.3)
    out = highway(inputs)
    expected_out_shape = (5, 100)
    assert(torch.Size(expected_out_shape) == out.shape), "The shape of Highway output is incorrect" 

    """ Matrix Mult for Highway Class """

    def reinitialize_layers(model):
        """ Reinitialize the Layer Weights for Sanity Checks.
        """
        def init_weights(m):
            if type(m) == nn.Linear:
                m.weight.data = torch.tensor([[.1, .1, .1, .1, .1], [.03, .03, .03, .03, .03],[.5, .5, .5, .5, .5], [-.7, -.7, -.7,- .7, -.7], [-.9, -.9, -.9, -.9, -.9]] )
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif type(m) == nn.Embedding:
                m.weight.data.fill_(0.15)
            elif type(m) == nn.Dropout:
                nn.Dropout(DROPOUT_RATE)
        with torch.no_grad():
            model.apply(init_weights)

    highway = Highway(5, 0)
    reinitialize_layers(highway)
    inputs =torch.tensor([[3, 5, 7, 11, 13]], dtype=torch.float) # Sums to 39
    # W * x + b = [3.9, 1.18, 19.6, -27.4, -35.2] 
    # relu (w x + b) =  [3.9, 1.18, 19.6, 0, 0] 
    # sigmoid(w x + b) = [.98, 0.76, 1, 0, 0]
    # sgimoid * relu(wx + b) + (1- sigmoid) * x = [3.9, 2.09, 19.6, 11, 13]
    ans = highway(inputs)
    expected = np.array([3.9, 2.09, 19.6, 11, 13])
    assert(np.allclose(expected, ans.detach().numpy(), atol = 0.1)), "highway function not giving expected output"

    print("Sanity Check Passed for Question 1f: Highway Class")
    print("-"*80)


def question_1g_sanity_check():
    """ Sanity check for CNN() class.
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1g: CNN Class")


    """ Shape Dimension Check for CNN Class """
    batch_size = 6

    e_char_size = 20
    num_filters = 10
    k = 5
    m_word_size = 100

    e_word_size = num_filters

    inputs = torch.randint(100,size=(batch_size, e_char_size, m_word_size), dtype=torch.float)
    convnet = CNN(20, 10, 5, 1)
    out = convnet(inputs)
    expected_out_shape = (batch_size, e_word_size)


    assert(torch.Size(expected_out_shape) == out.shape), "The shape of CNN output is incorrect" 


    """ Hand Calculate Check for CNN Class """
    def reinitialize_layers(model):
        """ Reinitialize the Layer Weights for Sanity Checks.
        """
        def init_weights(m):
            if type(m) == nn.Conv1d:
                m.weight.data = torch.tensor([[[0.1, .1, .1, .1, .5]]])
                if m.bias is not None:
                    m.bias.data.fill_(-0.5)
        with torch.no_grad():
            model.apply(init_weights)

    batch_size = 1
    m_word_size = 6
    e_char_size = 1
    num_filters = 1
    k = 5
    e_word_size = num_filters

    inputs = torch.tensor([[[1, 1, 1, 1, 0, 1]]], dtype=torch.float)
    convnet = CNN(e_char_size, num_filters, k, 0)
    reinitialize_layers(convnet)
    out = convnet(inputs)
    expected = np.array([[0.3]])
    assert(np.allclose(expected, out.detach().numpy(), atol = 0.1)), "CNN network not giving expected output for simple test case 1"


    inputs2 = torch.tensor([[[-1, -1, -1, -1, 0, -1]]], dtype=torch.float)
    out2 = convnet(inputs2)
    expected2 = np.array([[0]])
    assert(np.allclose(expected2, out2.detach().numpy(), atol = 0.1)), "CNN network giving expected output for simple test case 2"

    print("Sanity Check Passed for Question 1g: CNN Class")
    print("-"*80)


def question_1h_sanity_check(model):
    """ Sanity check for model_embeddings.py
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 1h: Model Embedding")
    print ("-"*80)
    sentence_length = 10
    max_word_length = 21
    inpt = torch.zeros(sentence_length, BATCH_SIZE, max_word_length, dtype=torch.long)
    ME_source = model.model_embeddings_source
    output = ME_source.forward(inpt)
    output_expected_size = [sentence_length, BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    print("Sanity Check Passed for Question 1h: Model Embedding!")
    print("-"*80)

def question_2a_sanity_check(decoder, char_vocab):
    """ Sanity check for CharDecoder.forward()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2a: CharDecoder.forward()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    logits, (dec_hidden1, dec_hidden2) = decoder.forward(inpt)
    logits_expected_size = [sequence_length, BATCH_SIZE, len(char_vocab.char2id)]
    dec_hidden_expected_size = [1, BATCH_SIZE, HIDDEN_SIZE]
    assert(list(logits.size()) == logits_expected_size), "Logits shape is incorrect:\n it should be {} but is:\n{}".format(logits_expected_size, list(logits.size()))
    assert(list(dec_hidden1.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden1.size()))
    assert(list(dec_hidden2.size()) == dec_hidden_expected_size), "Decoder hidden state shape is incorrect:\n it should be {} but is: {}".format(dec_hidden_expected_size, list(dec_hidden2.size()))
    print("Sanity Check Passed for Question 2a: CharDecoder.forward()!")
    print("-"*80)

def question_2b_sanity_check(decoder):
    """ Sanity check for CharDecoder.train_forward()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2b: CharDecoder.train_forward()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(sequence_length, BATCH_SIZE, dtype=torch.long)
    loss = decoder.train_forward(inpt)
    assert(list(loss.size()) == []), "Loss should be a scalar but its shape is: {}".format(list(loss.size()))
    print("Sanity Check Passed for Question 2b: CharDecoder.train_forward()!")
    print("-"*80)

def question_2c_sanity_check(decoder):
    """ Sanity check for CharDecoder.decode_greedy()
        basic shape check
    """
    print ("-"*80)
    print("Running Sanity Check for Question 2c: CharDecoder.decode_greedy()")
    print ("-"*80)
    sequence_length = 4
    inpt = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)
    initialStates = (inpt, inpt)
    device = decoder.char_output_projection.weight.device
    decodedWords = decoder.decode_greedy(initialStates, device)
    assert(len(decodedWords) == BATCH_SIZE), "Length of decodedWords should be {} but is: {}".format(BATCH_SIZE, len(decodedWords))
    print("Sanity Check Passed for Question 2c: CharDecoder.decode_greedy()!")
    print("-"*80)

def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

    # Create NMT Model
    model = NMT(
        word_embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    char_vocab = DummyVocab()

    # Initialize CharDecoder
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)

    if args['1e']:
        question_1e_sanity_check()
    elif args['1f']:
        question_1f_sanity_check()
    elif args['1g']:
        question_1g_sanity_check()
    elif args['1h']:
        question_1h_sanity_check(model)
    elif args['2a']:
        question_2a_sanity_check(decoder, char_vocab)
    elif args['2b']:
        question_2b_sanity_check(decoder)
    elif args['2c']:
        question_2c_sanity_check(decoder)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
