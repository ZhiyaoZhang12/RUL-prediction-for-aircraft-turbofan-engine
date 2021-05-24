import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torch.optim as optim
from Trainer import Trainer


SEED = 1337

class RNNModel(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=10,
                 num_layers=1,
                 hidden_size=10,
                 cell_type='LSTM',
                 dropout_train = 0.5):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_cell = None
        self.cell_type = cell_type
        self.dropout_train = dropout_train
        self.output_size = output_size
        self.kernel_size = kernel_size

        assert self.cell_type in ['LSTM', 'RNN', 'GRU', 'QRNN', 'TCN', 'DRNN'], \
            'Not Implemented, choose on of the following options - ' \
            'LSTM, RNN, GRU'

        if self.cell_type == 'LSTM':
            self.encoder_cell = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        if self.cell_type == 'GRU':
            self.encoder_cell = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout = self.dropout_train, bidirectional = False)
        if self.cell_type == 'RNN':
            self.encoder_cell = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)


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


class RNNTrainer(Trainer):
    def __init__(self,
                 lr,
                 max_lr,
                 total_steps,
                 number_steps_train,
                 hidden_size,
                 num_layers,
                 cell_type,
                 dropout_train,
                 batch_size,
                 num_epoch,
                 number_features_input=1,
                 number_features_output=1,
                 kernel_size=None,
                 loss_function='MSE',
                 optimizer='Adam',
                 normalizer='Standardization',
                 use_scheduler=False,
                 validation_split=0.2,
                 **kwargs):

        super(RNNTrainer, self).__init__(**kwargs)

        torch.manual_seed(SEED)

        # Hyper-parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.dropout_train = dropout_train
        self.number_features_input = number_features_input
        self.number_features_output = number_features_output
        self.number_steps_train = number_steps_train
        self.lr = lr
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.use_scheduler = use_scheduler
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.validation_split = validation_split
        
        
        self.file_name = self.filelogger.file_name

        # Save metadata model
        metadata_key = ['number_steps_train',
                        'cell_type',
                        'dropout_train',
                        'hidden_size',
                        'kernel_size',
                        'num_layers',
                        'lr',
                        'batch_size',
                        'num_epoch']

        metadata_value = [self.number_steps_train,
                          self.cell_type,
                          self.dropout_train,
                          self.hidden_size,
                          self.kernel_size,
                          self.num_layers,
                          self.lr,
                          self.batch_size,
                          self.num_epoch]

        metadata_dict = {}
        for i in range(len(metadata_key)):
            metadata_dict[metadata_key[i]] = metadata_value[i]


        # check if it's to load model or not
        if self.filelogger.load_model is not None:
            self.load(self.filelogger.load_model)
            print('Load model from {}'.format(
                self.logger_path + self.file_name + 'model_checkpoint/' + self.filelogger.load_model))
        
            ### save path for already trained models 
            print('there are models well-trained')
            
        else:
            print('RNNModel training')        
            self.model = RNNModel(self.number_features_input,
                                  self.number_features_output,
                                  self.kernel_size,
                                  self.num_layers,
                                  self.hidden_size,
                                  self.cell_type,
                                  self.dropout_train)

            print(metadata_dict)
            self.filelogger.write_metadata(metadata_dict)

        # loss function
        if loss_function == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_function == 'SCORE':
            self.criterion = self.score_loss
        elif loss_function == 'MSE_SCORE':
            self.criterion = self.mse_score_loss

        # optimizer
        if optimizer == 'Adam':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'SGD':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif optimizer == 'RMSProp':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer == 'Adadelta':
            self.model_optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        elif optimizer == 'Adagrad':
            self.model_optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)

        if self.use_scheduler==1:
            self.scheduler = ReduceLROnPlateau(self.model_optimizer, 'min', patience=2, threshold=1e-5)
        elif self.use_scheduler==2:
            self.scheduler = CyclicLR(self.model_optimizer, self.lr, max_lr, step_size_up=1000, cycle_momentum=False)
        elif self.use_scheduler==3:
            self.scheduler = OneCycleLR(self.model_optimizer, max_lr, 
                                        total_steps=total_steps,
                                        cycle_momentum=False)

        # check CUDA availability
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        
        self.model.to(self.device)
    
    def score_loss(self, y_pred, y_true, alpha1=13, alpha2=10):
        batch_size = y_pred.size(0)
        e = y_pred - y_true
        e = e.view(-1, 1).clamp(-100, 100)
        e1 = e[e<=0]
        e2 = e[e>0]
        s1 = (torch.exp(-(e1 / alpha1)) - 1)
        s2 = (torch.exp((e2 / alpha2)) - 1)
        score = torch.cat((s1, s2))
        score = score[torch.isfinite(score)].mean()
#         print(s1)
#         print(s2)
#         print(score)
#         loss = score if torch.isfinite(score) else F.mse_loss(y_pred, y_true)
        return score
    
        # loss function combining MSE and SCORE   
    def mse_score_loss(self, y_pred,y_true,alpha1=13,alpha2=10):
        batch_size = y_pred.size(0)
        e = y_pred - y_true
        e = e.view(-1, 1).clamp(-100, 100)
        e1 = e[e<=0]
        e2 = e[e>0]
        s1 = (torch.exp(-(e1 / alpha1)) - 1)
        s2 = (torch.exp((e2 / alpha2)) - 1)
        score = torch.cat((s1, s2))
        score = score[torch.isfinite(score)].mean()
        mse = F.mse_loss(y_pred, y_true)
        mse_score = F.mse_loss(y_pred, y_true) +score*8
        
        return mse_score
    
    def training_step(self):

        self.model_optimizer.zero_grad()
        loss = 0
        X, Y = next(self.datareader.train_generator)
        length = X.shape[0]
        X = torch.from_numpy(X).float().to(self.device)
        Y = torch.from_numpy(Y).float()

        results = self.model(X)
#         print(results.shape, Y.shape)
#         loss = self.criterion(results, Y.unsqueeze(2).to(self.device))
        loss = self.criterion(results, Y.to(self.device))

        loss.backward()
        self.model_optimizer.step()

        return loss.item(), loss.item() * length

    def evaluation_step(self):

        X, Y = next(self.datareader.validation_generator)
        length = X.shape[0]
        X = torch.from_numpy(X).float().to(self.device)
        Y = torch.from_numpy(Y).float().to(self.device)

        results = self.model.predict(X)

#         valid_loss = self.criterion(results, Y.unsqueeze(2).to(self.device))
        valid_loss = self.criterion(results, Y.to(self.device))

        return valid_loss.item(), valid_loss.item() * length

    def prediction_step(self):

        X, Y = next(self.datareader.test_generator)
        X = torch.from_numpy(X).float().to(self.device)

        results = self.model.predict(X)

        return results, Y
    