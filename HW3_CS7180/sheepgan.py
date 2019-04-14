""" Predict sheep sketches 

This initial model is just me reimplementing an LSTM
from HW2 then trying to get that working. The Sketch-RNN
looks interesting but I don't want to try that until I 
have all the required features implemented.
"""
import logging

import numpy as np

import torch
import torch.nn as nn

from logging_utils import enable_cloud_log


logger = logging.getLogger(__name__)


class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        logger.info("input: {}, hidden: {}".format(input.shape, hidden.shape))
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input)
        logger.info("input: {}, hidden: {}".format(input.shape, hidden.shape))
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


def train_sheepgan():
    pass

