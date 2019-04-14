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
from torch import optim
import torch.nn as nn

from logging_utils import enable_cloud_log


logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


############################################# hyperparameters

class SharedHParams(object):

    max_seq_length = 200
    train_set = 'Sheep_Market/train.npy'
    test_set = 'Sheep_Market/test.npy'
    val_set = 'Sheep_Market/valid.npy'
    learning_rate = 0.005
    batch_size = 100
    Nmax = 0
    gen_hidden_size = 256

class GeneratorHParams(SharedHParams):
    input_size = 1
    hidden_size = 200
    output_size = 1
    n_layers = 1

class DiscriminatorHParams(SharedHParams):
    input_size = 1
    hidden_size = 1
    output_size = 1
    n_layers = 1

############################################ load and prepare data

class SketchDataPipeline(object):
    """ Data pipeline is based on reference (1) """

    def __init__(self, hparams, data_location):
        self.hp = hparams
        self.data_location = data_location

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

    def make_batch(self, data, batch_size, Nmax):
        batch_idx = np.random.choice(len(data),batch_size)
        batch_sequences = [data[idx] for idx in batch_idx]
        strokes = []
        lengths = []
        indice = 0
        for seq in batch_sequences:
            len_seq = len(seq[:,0])
            new_seq = np.zeros((Nmax,5))
            new_seq[:len_seq,:2] = seq[:,:2]
            new_seq[:len_seq-1,2] = 1-seq[:-1,2]
            new_seq[:len_seq,3] = seq[:,2]
            new_seq[(len_seq-1):,4] = 1
            new_seq[len_seq-1,2:4] = 0
            lengths.append(len(seq[:,0]))
            strokes.append(new_seq)
            indice += 1

        if use_cuda:
            batch = torch.from_numpy(np.stack(strokes,1)).\
                type(torch.LongTensor).cuda()
        else:
            batch = torch.from_numpy(np.stack(strokes,1)).\
                type(torch.LongTensor)
        return batch, lengths

    def get_clean_batch(self):
        """ Execute data pipeline on a data location """
        data = np.load(self.data_location, encoding='latin1')
        data = self.purify(data)
        data = self.normalize(data)
        self.hp.Nmax = self.max_size(data)

        batch, lengths = self.make_batch(data, self.hp.batch_size,
                                         self.hp.Nmax)
        return batch, lengths, self.hp


class Generator(nn.Module, GeneratorHParams):

    def __init__(self):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(5, self.gen_hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        logger.info("input: {}, hidden: {}".\
                    format(len(input), len(hidden)))

        output, hidden = self.lstm(input, hidden)
        return output, hidden

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

class SheepModelV0(object):

    __name__ = "SheepModelV0"

    def __init__(self, shared_hparams, data_pipeline, criterion):

        self.shp = shared_hparams
        self.dp = data_pipeline
        self.criterion = criterion

        if use_cuda:
            self.generator = Generator().cuda()
            #self.discriminator = Discriminator().cuda()
        else:
            self.generator = Generator()
            
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), self.shp.learning_rate
        )
        #self.discriminator_optimizer = optim.Adam(
        #    self.discriminator.parameters(), self.shp.learning_rate
        #)

    def train(self, epoch):

        # Init generator for epoch
        
        self.generator.zero_grad()
        self.generator.train()

        # Init discriminator for epoch

        #disc_hidden = self.discriminator.init_hidden()
        #disc_hidden.zero_grad()
        #disc_hidden.train()

        # Fetch batches from data pipeline
        
        batch, lengths, hp = self.dp.get_clean_batch()
        self.shp = hp

        # Input and target
        # looks like batch is the target

        # Process batch

        print(type(batch))
        print(type(lengths[0]))
        
        gen_output, gen_hidden = self.generator(batch, lengths)

        # Prepare optimizers

        self.generator_optimizer.zero_grad()

        # Compute losses

        loss = self.criterion(gen_output, batch)

        # Gradient step

        loss.backward()

        # Optimizer step

        self.generator_optimizer.step()

        # Log progress

        if epoch % 1 == 0:
            logger.info("epoch {}".format(epoch))

            

############################################# train generic model


def train_model(num_epochs: int, model):
    """ Based on HW2, Rose on the white board, and reference (2) """

    logger.info("STARTED training {}".format(model.__name__))

    for epoch in range(num_epochs):
        
        logger.debug("Started epoch: {}".format(epoch))
        model.train(epoch)

        if epoch % 10 == 0:
            logger.info("Completed epoch: {}".format(epoch))

    logger.info("FINISHED training {}".format(model.__name__))


############################################# run this script

if __name__ == "__main__":

    shared_hparams = SharedHParams() 
    data_pipeline = SketchDataPipeline(
        shared_hparams,
        shared_hparams.train_set
    )
    criterion = nn.CrossEntropyLoss()
    
    
    model = SheepModelV0(shared_hparams, data_pipeline, criterion)
    train_model(num_epochs=1, model=model)
    
