"""
Using Sketch-RNN as a basis for my SheepGAN

Seq2Seq Autoencoders have some nice properties for this
type of data. Let's see if I can extend the approach to 
play two models off against each other. Could be a lot of
fun.

This implementation of Sketch-RNN is from
https://github.com/alexis-jacq/Pytorch-Sketch-RNN

I've updated some minor parts for Pytorch API compatibility 
(`torch.__version__ == '1.0.1.post2'`) and the sheep dataset 
we were given for HW3. I've also modified Sketch-RNN so it works
as a generator for the GAN.

References:
  (1) https://github.com/alexis-jacq/Pytorch-Sketch-RNN
  (2) https://github.com/eriklindernoren/PyTorch-GAN
  (3) https://github.com/j-min/Adversarial_Video_Summary
"""
import logging

import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


############################################################ Hyperparameters

class SharedHParams(object):

    max_seq_length = 200
    train_set = 'Sheep_Market/train.npy'
    test_set = 'Sheep_Market/test.npy'
    val_set = 'Sheep_Market/valid.npy'
    batch_size = 100
    Nmax = 0
    
    lr = 0.001
    lr_decay = 0.9999
    min_lr = 0.00001


class GeneratorHParams(SharedHParams):

    enc_hidden_size = 256
    dec_hidden_size = 512
    Nz = 128
    M = 20
    dropout = 0.9
    eta_min = 0.01
    R = 0.99995
    KL_min = 0.2
    wKL = 0.5
    grad_clip = 1.
    temperature = 0.4
    max_seq_length = 200

class DiscriminatorHParams(SharedHParams):

    enc_hidden_size = 256
    dec_hidden_size = 512
    Nz = 128
    M = 20
    dropout = 0.9
    eta_min = 0.01
    R = 0.99995
    KL_min = 0.2
    wKL = 0.5
    grad_clip = 1.
    temperature = 0.4
    max_seq_length = 200


############################################ load and prepare data

class SketchDataPipeline(object):
    """ Data pipeline is based on reference (1) """

    def __init__(self, hparams, data_location):
        self.hp = hparams
        self.data_location = data_location
        self.data, self.Nmax = self.get_clean_data()

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

    def make_batch(self, batch_size):

        data = self.data
        Nmax = self.Nmax
        
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
                type(torch.FloatTensor).cuda()
        else:
            batch = torch.from_numpy(np.stack(strokes,1)).\
                type(torch.FloatTensor)
        return batch, lengths

    def get_clean_data(self):
        """ Execute data pipeline on a data location """
        data = np.load(self.data_location, encoding='latin1')
        data = self.purify(data)
        data = self.normalize(data)
        Nmax = self.max_size(data)
        return data, Nmax

#################################################
# Sequence-to-Sequence Variational AutoEncoder
#
# Using variations of this base model as a
# generator and discriminator for the GAN
#################################################


class EncoderRNN(nn.Module):
    def __init__(self, vhp):
        super(EncoderRNN, self).__init__()

        # Hyper parameters

        self.vhp = vhp
        
        # Data pipeline
        
        self.sdp = SketchDataPipeline(self.vhp, self.vhp.train_set)
        
        # bidirectional lstm:
        
        self.lstm = nn.LSTM(5, self.vhp.enc_hidden_size, \
            dropout=self.vhp.dropout, bidirectional=True)
        
        # create mu and sigma from lstm's last output:
        
        self.fc_mu = nn.Linear(2*self.vhp.enc_hidden_size, self.vhp.Nz)
        self.fc_sigma = nn.Linear(2*self.vhp.enc_hidden_size, self.vhp.Nz)
        
        # active dropout:
        
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            
            # then must init with zeros
            
            if use_cuda:
                hidden = torch.zeros(2, batch_size,
                                     self.vhp.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size,
                                   self.vhp.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size, self.vhp.enc_hidden_size)
                cell = torch.zeros(2, batch_size, self.vhp.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden,cell) = self.lstm(inputs.float(), hidden_cell)
        
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        
        hidden_forward, hidden_backward = torch.split(hidden,1,0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward\
                                .squeeze(0)],1)
        # mu and sigma:
        
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        
        # N ~ N(0,1)
        
        z_size = mu.size()
        if use_cuda:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size))
        z = mu + sigma*N
        
        # mu and sigma_hat are needed for LKL loss
        
        return z, mu, sigma_hat


class DecoderRNN(nn.Module):
    def __init__(self, vhp):
        super(DecoderRNN, self).__init__()

        self.vhp = vhp

        # Data Pipeline
        
        self.sdp = SketchDataPipeline(self.vhp, self.vhp.train_set)

        # to init hidden and cell from z:
        
        self.fc_hc = nn.Linear(self.vhp.Nz, 2*self.vhp.dec_hidden_size)
        
        # unidirectional lstm:
        
        self.lstm = nn.LSTM(self.vhp.Nz+5, self.vhp.dec_hidden_size,
                            dropout=self.vhp.dropout)
        
        # create proba distribution parameters from hiddens:
        
        self.fc_params = nn.Linear(self.vhp.dec_hidden_size,6*self.vhp.M+3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            
            # then we must init from z
            
            hidden,cell = torch.split(F.tanh(self.fc_hc(z)),
                                      self.vhp.dec_hidden_size,1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(),
                           cell.unsqueeze(0).contiguous())
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        
        if self.training:
            y = self.fc_params(outputs.view(-1, self.vhp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.vhp.dec_hidden_size))
            
        # separate pen and mixture params:
        
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        
        # identify mixture params:
        
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2)
        
        # preprocess params::
        
        if self.training:
            len_out = self.sdp.Nmax+1
        else:
            len_out = 1
                                   
        pi = F.softmax(pi.transpose(0,1).squeeze()).\
            view(len_out,-1,self.vhp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).\
            view(len_out,-1,self.vhp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).\
            view(len_out,-1,self.vhp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).\
            view(len_out,-1,self.vhp.M)
        mu_x = mu_x.transpose(0,1).squeeze().contiguous().\
            view(len_out,-1,self.vhp.M)
        mu_y = mu_y.transpose(0,1).squeeze().contiguous().\
            view(len_out,-1,self.vhp.M)
        q = F.softmax(params_pen).view(len_out,-1,3)
        return pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy,q,hidden,cell


class Sequence2SequenceVaeUtils(object):

    def __init__(self, vhp, sdp):
        self.vhp = vhp
        self.sdp = sdp

        self.eta_step = self.vhp.eta_min

    ## Local hyperparams

    # Encoder output
    mu = None
    sigma = None

    # Decoder output
    pi = None
    mu_x = None
    mu_y = None
    sigma_x = None
    sigma_y = None
    rho_xy = None
    q = None

    def ler_decay(self, optimizer):
        """Decay learning rate by a factor of lr_decay

        Adaptive learning rate, see reference (1)
        """
        for param_group in optimizer.param_groups:
            if param_group['lr']>self.vhp.min_lr:
                param_group['lr'] *= self.vhp.lr_decay
        return optimizer

    def make_target(self, batch, lengths):
        if use_cuda:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.\
                              size()[1]).cuda().unsqueeze(0)
        else:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.\
                              size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.sdp.Nmax+1, batch.size()[1])
        for indice,length in enumerate(lengths):
            mask[:length,indice] = 1
        if use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:,:,0]]*self.vhp.M,2)
        dy = torch.stack([batch.data[:,:,1]]*self.vhp.M,2)
        p1 = batch.data[:,:,2]
        p2 = batch.data[:,:,3]
        p3 = batch.data[:,:,4]
        p = torch.stack([p1,p2,p3],2)
        return mask,dx,dy,p

    
    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask*torch.log(1e-5+torch.sum(self.pi * pdf, 2)))\
            /float(self.sdp.Nmax*ghp.batch_size)
        LP = -torch.sum(p*torch.log(self.q))/\
            float(self.sdp.Nmax*ghp.batch_size)
        return LS+LP

    def kullback_leibler_loss(self):
        LKL = -0.5*torch.sum(1+self.sigma-self.mu**2-torch.exp(self.sigma))\
            /float(ghp.Nz*ghp.batch_size)
        if use_cuda:
            KL_min = torch.Tensor([ghp.KL_min]).cuda().detach()
        else:
            KL_min = torch.Tensor([ghp.KL_min]).detach()
        return self.vhp.wKL*self.eta_step * torch.max(LKL,KL_min)

    def conditional_generation(self, epoch, encoder, decoder):
        batch,lengths = self.sdp.make_batch(1)
        # should remove dropouts:
        encoder.train(False)
        decoder.train(False)
        # encode:
        z, _, _ = encoder(batch, 1)
        if use_cuda:
            sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1).cuda()
        else:
            sos = torch.Tensor([0,0,1,0,0]).view(1,1,-1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.sdp.Nmax):
            input = torch.cat([s,z.unsqueeze(0)],2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            #------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print(i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample,y_sample,z_sample]).T
        self.make_image(sequence, epoch)

    def sample_next_state(self):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/ghp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0,0,:].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(ghp.M, p=pi)
        # get pen state:
        q = self.q.data[0,0,:].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0,0,pi_idx]
        mu_y = self.mu_y.data[0,0,pi_idx]
        sigma_x = self.sigma_x.data[0,0,pi_idx]
        sigma_y = self.sigma_y.data[0,0,pi_idx]
        rho_xy = self.rho_xy.data[0,0,pi_idx]
        x,y = self.sample_bivariate_normal(mu_x,mu_y,sigma_x,
                                           sigma_y,rho_xy,greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        if use_cuda:
            return next_state.cuda().view(1,1,-1),x,y,q_idx==1,q_idx==2
        else:
            return next_state.view(1,1,-1),x,y,q_idx==1,q_idx==2

    def sample_bivariate_normal(self, mu_x,mu_y,sigma_x,sigma_y,
                                rho_xy, greedy=False):
        # inputs must be floats
        if greedy:
            return mu_x,mu_y
        mean = [mu_x, mu_y]
        sigma_x *= np.sqrt(ghp.temperature)
        sigma_y *= np.sqrt(ghp.temperature)
        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
               [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def make_image(self, sequence, epoch, name='_output_'):
        """plot drawing with separated strokes"""
        strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for s in strokes:
            plt.plot(s[:,0],-s[:,1])
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                        canvas.tostring_rgb())
        name = str(epoch)+name+'.jpg'
        fpath = "outputs/" + name
        pil_image.save(fpath,"JPEG")
        plt.close("all")


############################################################ Generator


class Generator(Sequence2SequenceVaeUtils):
    """ Generator model, based on Sketch-RNN from Reference (1) """

    __name__ = "Generator"
    
    def __init__(self, vhp):

        # Alias for hyper params b/c of inheritance

        self.vhp = vhp
        self.ghp = self.vhp

        # Setting up generator
        
        if use_cuda:
            self.encoder = EncoderRNN(self.ghp).cuda()
            self.decoder = DecoderRNN(self.ghp).cuda()
        else:
            self.encoder = EncoderRNN(self.ghp)
            self.decoder = DecoderRNN(self.ghp)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                            self.ghp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            self.ghp.lr)
        self.eta_step = ghp.eta_min
        self.sdp = SketchDataPipeline(self.ghp, self.ghp.train_set)

        ## Local hyperparams

        # Encoder output
        self.mu = None
        self.sigma = None

        # Decoder output
        self.pi = None
        self.mu_x = None
        self.mu_y = None
        self.sigma_x = None
        self.sigma_y = None
        self.rho_xy = None
        self.q = None

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths = self.sdp.make_batch(self.ghp.batch_size)
        # encode:
        z, self.mu, self.sigma = self.encoder(batch, self.ghp.batch_size)
        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*\
                              self.ghp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*\
                              self.ghp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch],0)
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z]*(self.sdp.Nmax+1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack],2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        # prepare targets:
        mask,dx,dy,p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1-(1-ghp.eta_min)*self.ghp.R
        # compute losses:
        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask,dx,dy,p,epoch)
        loss = LR + LKL
        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm(self.encoder.parameters(),
                                self.ghp.grad_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(),
                                self.ghp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch%1==0:
            logger.info("{} - epoch {}, loss {}, LR {}, LKL {}".\
                        format(self.__name__, epoch, loss.data.item(),
                               LR.data.item(), LKL.data.item()))
            self.encoder_optimizer = self.ler_decay(self.encoder_optimizer)
            self.decoder_optimizer = self.ler_decay(self.decoder_optimizer)
        if epoch%100==0:
            #self.save(epoch)
            self.conditional_generation(epoch, self.encoder, self.decoder)

    def save(self, epoch):
        sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
                   'encoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
                   'decoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)


############################################################ Discriminator


class Discriminator(Sequence2SequenceVaeUtils):
    """ Discriminator model, based on Sketch-RNN from Reference (1) """

    __name__ = "Discriminator"

    def __init__(self, vhp):

        # Alias for hyper params b/c of inheritance

        self.vhp = vhp
        self.dhp = self.vhp

        # Setting up generator
        
        if use_cuda:
            self.encoder = EncoderRNN(self.dhp).cuda()
            self.decoder = DecoderRNN(self.dhp).cuda()
        else:
            self.encoder = EncoderRNN(self.dhp)
            self.decoder = DecoderRNN(self.dhp)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                            self.dhp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            self.dhp.lr)
        self.eta_step = dhp.eta_min
        self.sdp = SketchDataPipeline(self.dhp, self.dhp.train_set)

        ## Local hyperparams

        # Encoder output
        self.mu = None
        self.sigma = None

        # Decoder output
        self.pi = None
        self.mu_x = None
        self.mu_y = None
        self.sigma_x = None
        self.sigma_y = None
        self.rho_xy = None
        self.q = None

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths = self.sdp.make_batch(self.dhp.batch_size)
        # encode:
        z, self.mu, self.sigma = self.encoder(batch, self.dhp.batch_size)
        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*\
                              self.dhp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*\
                              self.dhp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch],0)
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z]*(self.sdp.Nmax+1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack],2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        # prepare targets:
        mask,dx,dy,p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1-(1-self.dhp.eta_min)*self.dhp.R
        # compute losses:
        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask,dx,dy,p,epoch)
        loss = LR + LKL
        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm(self.encoder.parameters(),
                                self.dhp.grad_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(),
                                self.dhp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch%1==0:
            logger.info("{} - epoch {}, loss {}, LR {}, LKL {}".\
                        format(self.__name__, epoch, loss.data.item(),
                               LR.data.item(), LKL.data.item()))
            self.encoder_optimizer = self.ler_decay(self.encoder_optimizer)
            self.decoder_optimizer = self.ler_decay(self.decoder_optimizer)
        if epoch%100==0:
            #self.save(epoch)
            self.conditional_generation(epoch, self.encoder, self.decoder)

    def save(self, epoch):
        sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
                   'encoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))
        torch.save(self.decoder.state_dict(), \
                   'decoderRNN_sel_%3f_epoch_%d.pth' % (sel,epoch))

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)



############################################################ SheepGAN

class SheepGAN(object):
    """ 
    Game plan is to get two Sequence-to-Sequence Variational Autoencoders
    to play ball.
    """

    def __init__(self, num_epochs: int, ghp, dhp):
        self.num_epochs = num_epochs
        self.ghp = ghp
        self.dhp = dhp

    def train(self):
        logger.info("STARTED training SheepGAN")

        generator = Generator(vhp=self.ghp)
        discriminator = Discriminator(vhp=self.dhp)

        for epoch in range(self.num_epochs):

            generator.train(epoch)
            discriminator.train(epoch)

            if epoch % 10 == 0:
                logger.info("Completed epoch: {}".format(epoch))

        logger.info("FINISHED training SheepGAN")


########################################################### Logging format

def enable_cloud_log(level='INFO'):
    """ Enable logs using default StreamHandler """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
    }
    logging.basicConfig(level=levels[level],
                        format='%(asctime)s %(levelname)s %(message)s')
        

############################################################ Training

if __name__ == "__main__":

    enable_cloud_log('DEBUG')
    ghp = GeneratorHParams()
    dhp = DiscriminatorHParams()
    bah = SheepGAN(num_epochs=5, ghp=ghp, dhp=dhp)
    bah.train()

