# -*- coding: utf-8 -*-
"""
cell_3: Class RNNModel(nn.Module)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch.optim as optim
#from turbofan_pkg.models.QRNN import QRNN
#from turbofan_pkg.models.TCN import TemporalConvNet
#from turbofan_pkg.models.DRNN import DRNN

SEED = 1337

class RNNModel(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=10,
                 num_layers=1,
                 hidden_size=10,
                 cell_type='LSTM'):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_cell = None
        self.cell_type = cell_type
        self.output_size = output_size
        self.kernel_size = kernel_size

        assert self.cell_type in ['LSTM', 'RNN', 'GRU', 'QRNN', 'TCN', 'DRNN'], \
            'Not Implemented, choose on of the following options - ' \
            'LSTM, RNN, GRU'

        if self.cell_type == 'LSTM':
            self.encoder_cell = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'GRU':
            self.encoder_cell = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout = dropout_train, bidirectional = False)
        if self.cell_type == 'RNN':
            self.encoder_cell = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
#        if self.cell_type == 'QRNN':
#            self.encoder_cell = QRNN(self.input_size, self.hidden_size, self.num_layers, self.kernel_size)
#        if self.cell_type == 'DRNN':
#            self.encoder_cell = DRNN(self.input_size, self.hidden_size, self.num_layers)  # Batch_First always True
#        if self.cell_type == 'TCN':
#            self.encoder_cell = TemporalConvNet(self.input_size, self.hidden_size, self.num_layers,
#                                                self.kernel_size)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden=None):
        outputs, hidden_state = self.encoder_cell(x,
                                                  hidden)  # returns output variable - all hidden states for seq_len, hindden state - last hidden state
        outputs = self.output_layer(outputs)

        return outputs

    def predict(self, x, hidden=None):

        prediction, _ = self.encoder_cell(x, hidden)
        prediction = self.output_layer(prediction)

        return prediction