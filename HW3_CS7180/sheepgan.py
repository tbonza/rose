""" Predict sheep sketches 

This initial model is just me reimplementing an LSTM
from HW2 then trying to get that working. The Sketch-RNN
looks interesting but I don't want to try that until I 
have all the required features implemented.

Code and higher level ideas have been used from my
references cited below.

References:
  (1) https://github.com/alexis-jacq/Pytorch-Sketch-RNN
  (2) https://github.com/eriklindernoren/PyTorch-GAN
"""
import logging

import numpy as np

import torch
import torch.nn as nn

from logging_utils import enable_cloud_log


logger = logging.getLogger(__name__)


############################################# hyperparameters

class SharedHParams(object):
    def __init__(self):
        self.max_seq_length = 200
        self.train_set = 'Sheep_Market/train.npy'
        self.test_set = 'Sheep_Market/test.npy'
        self.val_set = 'Sheep_Market/valid.npy'

class GeneratorHParams(SharedHParams):
    def __init__(self):
        self.input_size = 1
        self.hidden_size = 1
        self.output_size = 1
        self.n_layers = 1
        
class DiscriminatorHParams(SharedHParams):
    def __init__(self):
        self.input_size = 1
        self.hidden_size = 1
        self.output_size = 1
        self.n_layers = 1

############################################ load and prepare data

class SketchDataPipeline(object):
    """ Data pipeline is based on reference (1) """

    def __init__(self, hparms, data_location):
        self.hp = hparams
        self.data_location

    def max_size(self, data):
        """larger sequence length in the data set"""
        sizes = [len(seq) for seq in data]
        return max(sizes)

    def purify(self, strokes):
        """removes to small or too long sequences + removes large gaps"""
        data = []
        for seq in strokes:
            if seq.shape[0] <= self.hp.max_seq_length and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        return data

    def calculate_normalizing_scale_factor(self, strokes):
        """
        Calculate the normalizing factor explained in appendix of sketch-rnn.
        """
        data = []
        for i in range(len(strokes)):
            for j in range(len(strokes[i])):
                data.append(strokes[i][j, 0])
                data.append(strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, strokes):
        """
        Normalize entire dataset (delta_x, delta_y) by the scaling factor.
        """
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:2] /= scale_factor
            data.append(seq)
        return data

    def get_clean_data(self):
        """ Execute data pipeline on a data location """
        self.data = np.load(self.data_location, encoding='latin1')
        self.data = self.purify(self.data)
        self.data = self.normalize(self.data)

        return self.data

class Generator(nn.Module, GeneratorHParams):

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
        logger.info("input: {}, hidden: {}".\
                    format(input.shape, hidden.shape))
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

class Discriminator(nn.Module, DiscriminatorHParams):
    
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
        logger.info("input: {}, hidden: {}".\
                    format(input.shape, hidden.shape))
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


def train_sheepgan():
    #generator = Generator()
    #discriminator = Discriminator()

