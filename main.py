# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:15:10 2020
RUL predition with aircraft turbofan engine dataset

"""
'''
cell 1
'''
import os, datetime

# import hdbscan

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold

SEED = 1337

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

def error_function(df, y_predicted, y_true):
    return int(df[y_predicted] - df[y_true])

def score_function(df, label, alpha1=13, alpha2=10):
    if df[label] <= 0:
        return (np.exp(-(df[label] / alpha1)) - 1)

    elif df[label] > 0:
        return (np.exp((df[label] / alpha2)) - 1)

def accuracy_function(df, label, alpha1=13, alpha2=10):
    if df[label]<-alpha1 or df[label]>alpha2:
        return 0
    return 1

'''
cell2
'''
import sys, traceback, os, re

import torch
import numpy as np
import pandas as pd

from tqdm import trange

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.utils.tensorboard import SummaryWriter
from glob import glob

class Trainer(object):
    def __init__(self,
                 train_path,
                 test_path,
                 logger_path,
                 model_name,
                 train_log_interval,
                 valid_log_interval,
                 load_model_name=None,
                 use_script=False,
                 use_cuda = False,
                 **kwargs):

        # Data Reader
        self.datareader = DataReader(train_path,
                                     test_path,
                                     **kwargs)
        # File Logger
        self.filelogger = FileLogger(logger_path,
                                     model_name,
                                     load_model_name,
                                     use_script)

        # Check cuda availability
        self.use_cuda = torch.cuda.is_available() and use_cuda

        # Variables
        self.logger_path = logger_path
        self.model_name = model_name
        self.train_log_interval = train_log_interval
        self.valid_log_interval = valid_log_interval

    def save(self,
             model_name):
        """
        Save model
        """
        path = self.filelogger.path + '/model_checkpoint/'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model, path + model_name)

    def load(self,
             path_name):
        """
        Load model
        """
        print('Loading file from {}'.format(path_name))
        self.model = torch.load(path_name)
        if self.use_cuda:
            self.model.cuda()

    def train(self,
              patience):

        self.datareader.calculate_unique_turbines()
    
        self.datareader.prepare_datareader(self.batch_size, self.validation_split, self.number_steps_train, self.normalizer)

        self.filelogger.start()

        self.tensorboard = SummaryWriter(self.filelogger.path + '/tensorboard/')
        try:
            training_step = 0
            validation_step = 0

            best_validation_loss = 1000000000
            validation_loss = 1000000000
            train_loss = 1000000000
            best_validation_epoch = 0

            patience_step = 0

            epoch_range = trange(int(self.num_epoch),
                                  desc='1st loop',
                                  unit=' Epochs')

            for epoch in epoch_range:
#                 batch_train_range = trange(int(self.datareader.train_steps),
#                                             desc='2st loop',
#                                             unit=' Batch',
#                                             leave=True)

#                 batch_valid_range = trange(int(self.datareader.validation_steps),
#                                             desc='2st loop',
#                                             unit=' Batch',
#                                             leave=True)
                
                batch_train_range = range(int(self.datareader.train_steps))
                batch_valid_range = range(int(self.datareader.validation_steps))
                
                total_train_loss = 0

                for batch_train in batch_train_range:
#                     batch_train_range.set_description("Training on %i points --- " % self.datareader.train_length)

                    self.model.train()

                    loss, total_loss = self.training_step()

                    total_train_loss += total_loss

#                     batch_train_range.set_postfix(MSE=loss,
#                                                   Last_batch_MSE=train_loss,
#                                                   Epoch=epoch)

                    self.tensorboard.add_scalar('Training Mean Squared Error loss per batch',
                                                loss,
                                                training_step)

                    self.filelogger.write_train(self.train_log_interval,
                                                training_step,
                                                epoch,
                                                batch_train,
                                                loss)

                    training_step += 1

                train_loss = total_train_loss / (self.datareader.train_length)

                self.tensorboard.add_scalar('Training Mean Squared Error loss per epoch',
                                            train_loss,
                                            epoch)

                total_valid_loss = 0

                for batch_valid in batch_valid_range:
#                     batch_valid_range.set_description("Validate on %i points --- " % self.datareader.validation_length)

#                     batch_valid_range.set_postfix(Last_Batch_MSE=' {0:.9f} MSE'.format(validation_loss),
#                                                   Best_MSE=best_validation_loss,
#                                                   Best_Epoch=best_validation_epoch,
#                                                   Current_Epoch=epoch)

                    self.model.eval()

                    valid_loss, total_loss = self.evaluation_step()

                    total_valid_loss += total_loss

                    self.tensorboard.add_scalar('Validation Mean Squared Error loss per batch',
                                                valid_loss,
                                                validation_step)

                    self.filelogger.write_valid(self.valid_log_interval,
                                                validation_step,
                                                epoch,
                                                batch_valid,
                                                valid_loss)
                    validation_step += 1

                validation_loss = total_valid_loss / (self.datareader.validation_length)

                self.tensorboard.add_scalar('Validation Mean Squared Error loss per epoch',
                                            validation_loss,
                                            epoch)

                if self.use_scheduler==1:
                    self.scheduler.step(validation_loss)
                elif self.use_scheduler>=2:
                    self.scheduler.step()

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_epoch = epoch + 1
                    patience_step = 0
                    self.save('Model_Checkpoint' + str(epoch + 1) + '_valid_loss_' + str(best_validation_loss) + '.pth')
                else:
                    patience_step += 1
                    if patience_step > patience:
                        print('Train is donne, 3 epochs in a row without improving validation loss!')
                        
                        doc=open('D:/RUL/results/%s/2.txt'
                             %(dataset_name),'a')   
                    

                        print('best_validation_loss', best_validation_loss,file=doc)
                        doc.close()
                        
                        print("best_validation_loss have been saved")
        
                        return best_validation_loss

            print('Train is donne after 10 epochs!')
            return best_validation_loss

        except KeyboardInterrupt:
            if epoch > 0:
                print("Shutdown requested...saving and exiting")
                self.save('Model_save_before_exiting_epoch_' + str(epoch + 1) + '_batch_' + str(
                    batch_train) + '_batch_valid_' + str(batch_valid) + '.pth')
            else:
                print('Shutdown requested!')

        except Exception:
            if epoch > 0:
                self.save('Model_save_before_exiting_epoch_' + str(epoch + 1) + '_batch_' + str(
                    batch_train) + '_batch_valid_' + str(batch_valid) + '.pth')
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

    def train_cv(self, number_splits, days, patience):

        try:
            mean_score = []
            cv_train_indexes, cv_val_indexes = self.datareader.cross_validation_time_series(number_splits,
                                                                                            days,
                                                                                            self.test_date)

            for model_number in range(number_splits):

                self.filelogger.start('Fold_Number{0}'.format(model_number + 1))
                self.tensorboard = SummaryWriter(self.filelogger.path + '/tensorboard/')

                self.prepare_datareader_cv(cv_train_indexes[model_number],
                                           cv_val_indexes[model_number])

                training_step = 0
                validation_step = 0

                best_validation_loss = 1000
                validation_loss = 1000
                train_loss = 1000
                best_validation_epoch = 0

                patience_step = 0

                epoch_range = trange(int(self.num_epoch),
                                     desc='1st loop',
                                     unit=' Epochs',
                                     leave=True)

                for epoch in epoch_range:
                    batch_train_range = trange(int(self.datareader.train_steps),
                                               desc='2st loop',
                                               unit=' Batch',
                                               leave=False)

                    batch_valid_range = trange(int(self.datareader.validation_steps),
                                               desc='2st loop',
                                               unit=' Batch',
                                               leave=False)

                    total_train_loss = 0
                    for batch_train in batch_train_range:
                        batch_train_range.set_description("Training on %i points --- " % self.datareader.train_length)

                        self.model.train()

                        loss, total_loss = self.training_step()

                        total_train_loss += total_loss

                        batch_train_range.set_postfix(MSE=loss,
                                                      Last_batch_MSE=train_loss,
                                                      Epoch=epoch)

                        self.tensorboard.add_scalar('Training Mean Squared Error loss per batch',
                                                    loss,
                                                    training_step)

                        self.filelogger.write_train(self.train_log_interval,
                                                    training_step,
                                                    epoch,
                                                    batch_train,
                                                    loss)

                        training_step += 1

                    train_loss = total_train_loss / (self.datareader.train_length)

                    self.tensorboard.add_scalar('Training Mean Squared Error loss per epoch',
                                                train_loss,
                                                epoch)

                    total_valid_loss = 0

                    for batch_valid in batch_valid_range:
                        batch_valid_range.set_description(
                            "Validate on %i points --- " % self.datareader.validation_length)

                        batch_valid_range.set_postfix(Last_Batch_MSE=' {0:.9f} MSE'.format(validation_loss),
                                                      Best_MSE=best_validation_loss,
                                                      Best_Epoch=best_validation_epoch,
                                                      Current_Epoch=epoch)

                        self.model.eval()

                        valid_loss, total_loss = self.evaluation_step()

                        total_valid_loss += total_loss

                        self.tensorboard.add_scalar('Validation Mean Squared Error loss per batch',
                                                    valid_loss,
                                                    validation_step)

                        self.filelogger.write_valid(self.valid_log_interval,
                                                    validation_step,
                                                    epoch,
                                                    batch_valid,
                                                    valid_loss)

                        validation_step += 1

                    validation_loss = total_valid_loss / (self.datareader.validation_length)

                    self.tensorboard.add_scalar('Validation Mean Squared Error loss per epoch',
                                                validation_loss,
                                                epoch)

                    if self.use_scheduler==1:
                        self.scheduler.step(validation_loss)
                    elif self.use_scheduler>=2:
                        self.scheduler.step()

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_validation_epoch = epoch + 1
                        patience_step = 0
                        self.save(
                            'Model_Checkpoint' + str(epoch + 1) + '_valid_loss_' + str(best_validation_loss) + '.pth')
                    else:
                        patience_step += 1
                        if patience_step > patience:
                            break

                mean_score.append(best_validation_loss)

            return np.mean(mean_score)

        except KeyboardInterrupt:
            print('Shutdown requested!')

        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

    def predict(self):

        predictions = []
        labels = []

        for batch_test in range(self.datareader.test_steps):
            self.model.eval()

            prediction, Y = self.prediction_step()
            predictions.append(prediction.cpu().detach().numpy())
            labels.append(Y)

        return np.concatenate(predictions), np.concatenate(labels)

    def postprocess(self,
                predictions,
                labels):

        predictions = predictions[:, self.number_steps_train - 1, 0]
        labels = labels[:, self.number_steps_train - 1, 0]

        predictions = predictions - 0.5
        predictions = predictions.round()
        predictions[predictions < 0] = 0

        df_test = pd.DataFrame(index=self.datareader.test.index)
        
        df_test['True_RUL'] = labels
        df_test['Predicted_RUL'] = predictions
        df_test['Error'] = df_test.apply(lambda df: error_function(df, 'Predicted_RUL', 'True_RUL'), axis=1)
        df_test['Score'] = df_test.apply(lambda df: score_function(df, 'Error'), axis=1)

        df_temp = []
        for index in df_test.index.to_series().unique():
            df_temp.append(df_test.loc[index].iloc[-1][['True_RUL', 'Predicted_RUL', 'Error', 'Score']])

        results = pd.DataFrame(df_temp)

        mse = mean_squared_error(results['Predicted_RUL'], results['True_RUL'])
        mae = mean_absolute_error(results['Predicted_RUL'], results['True_RUL'])
        r2 = r2_score(results['True_RUL'], results['Predicted_RUL'])
        
        results['Accuracy'] = results.apply(lambda df: accuracy_function(df, 'Error'), axis=1)

        score = results['Score'].sum()

        return df_test, results, mse, mae, r2, score

    def get_best(self,load_or_save):

        files = glob(self.filelogger.path + '/model_checkpoint/*')

        best = 10000000
        for file in files:
            number = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', file)
#             print(number)
            result = float(number[-1])
        
        
            if result < best:
                best = result
                
                if load_or_save == 'load':
                    doc=open('D:/RUL/results/%s/2.txt'
                         %(dataset_name),'a')   

                    print('validation_loss', result, file=doc)
                    doc.close()
    #                print("validation_loss recored")
                    
                best_file = file

        if load_or_save == 'load':
            self.load(best_file)
        elif load_or_save == 'save':
            return(best_file)
        
'''
cell3
'''
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


class RNNTrainer(Trainer):
    def __init__(self,
                 lr,
                 max_lr,
                 total_steps,
                 number_steps_train,
                 hidden_size,
                 num_layers,
                 cell_type,
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
                        'hidden_size',
                        'kernel_size',
                        'num_layers',
                        'lr',
                        'batch_size',
                        'num_epoch']

        metadata_value = [self.number_steps_train,
                          self.cell_type,
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
        
            ### save path for the already trained model 
            print('we have already trained model')
            
        else:
            print('we do not have the already trained model, using RNNModel to train')        
            self.model = RNNModel(self.number_features_input,
                                  self.number_features_output,
                                  self.kernel_size,
                                  self.num_layers,
                                  self.hidden_size,
                                  self.cell_type)

            print(metadata_dict)
            self.filelogger.write_metadata(metadata_dict)

        # loss function
        if loss_function == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_function == 'SCORE':
            self.criterion = self.score_loss
        elif loss_function == 'MSE_SCORE':
            self.criterion = self.mse_score

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
    
    
    
    #combine the SCORE and the MSE to be a new loss function   
    def mse_score(self, y_pred,y_true,alpha1=13,alpha2=10):
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
        mse_score = F.mse_loss(y_pred, y_true)*10 +score*100
        
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
    
'''
cell4
'''
import os, json, shutil
import pandas as pd
import numpy as np


class FileLogger(object):
    def __init__(self,
                 path,
                 file_name,
                 model_name,
                 script):

        if script is not True:
            self.path = path + file_name + '/'
            self.load_model = None

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            else:
                while True:
                    overwrite = input('This file already exists. Do you want to overwrite it?'
                                      ' Y(yes) N(no) C(continue writing) E(exit)')
                    if overwrite == 'Y' or overwrite == 'N' or overwrite == 'C':
                        break
                    elif overwrite == 'E':
                        raise SystemExit
                    else:
                        print('Please choose one valid option. '
                              'Y for yes N for no or C for continue writing')

                if overwrite == 'N':
                    name = input('Choose a new name for the file:')
                    assert os.path.exists(path + name + '/') is False, \
                        'Yeah that one is already in use. Sorry dude! Please choose another name'
                    file_name = name
                    self.path = path + file_name + '/'
                    os.makedirs(self.path)
                    print('New directory created at {}'.format(self.path))

                if overwrite == 'Y':
                    while True:
                        overwrite = input('Are you completly fine with this. '
                                          'This will remove all previous files and existing models from this directory? Y(yes) N(no)')
                        if overwrite == 'Y' or overwrite == 'N':
                            break
                    if overwrite == 'Y':
                        shutil.rmtree(self.path)
                        os.makedirs(self.path)

                    else:
                        assert overwrite is None, \
                            'Well then you should check your files before.' \
                            ' You always can choose another file name to this model'

                if overwrite == 'C':
                    assert model_name is not None, \
                        'You didnt choose a model to resume train. ' \
                        'Please try again with a valid name or start a new session.'
                    assert os.path.exists(path + file_name + '/model_checkpoint/') is True, \
                        'You dont have models to resume. Please restart and start a new session'
                    assert os.path.exists(path + file_name + '/model_checkpoint/' + model_name) is True, \
                        'That model name dont exist. Please choose other model to resume'
                    self.load_model = path + file_name + '/model_checkpoint/' + model_name
        else:
            self.path = path + '/' + file_name
            self.file_path = self.path
            self.load_model = None

            if not os.path.exists(self.path):
                os.makedirs(self.path)
            else:
                print('removing')
                shutil.rmtree(self.path)
                os.makedirs(self.path)

        self.file_name = file_name

    def write_train(self,
                    log_interval,
                    step,
                    epoch,
                    batch,
                    loss):

        if batch % log_interval == 0:
            self.update_file(step,
                             epoch,
                             batch,
                             loss,
                             'train_log.txt')

    def write_valid(self,
                    log_interval,
                    step,
                    epoch,
                    batch,
                    loss):

        if batch % log_interval == 0:
            self.update_file(step,
                             epoch,
                             batch,
                             loss,
                             'valid_log.txt')

    def write_test(self,
                   log_interval,
                   step,
                   epoch,
                   batch,
                   loss):

        if batch % log_interval == 0:
            self.update_file(step,
                             epoch,
                             batch,
                             loss,
                             'test_log.txt')

    def write_metadata(self,
                       metadata):

        self.metadataLogger = open(self.path + '/metadata.txt', 'w')
        self.metadataLogger.write(json.dumps(metadata))
        self.metadataLogger.close()

    def open_writers(self):

        data = {'Step': [],
                'Epoch_Number': [],
                'Batch_number': [],
                'Loss': []
                }

        self.trainLogger = open(self.path + '/train_log.txt', 'w')
        self.trainLogger.write(json.dumps(data))
        self.trainLogger.close()
        self.validLogger = open(self.path + '/valid_log.txt', 'w')
        self.validLogger.write(json.dumps(data))
        self.validLogger.close()
        self.testLogger = open(self.path + '/test_log.txt', 'w')
        self.testLogger.write(json.dumps(data))
        self.testLogger.close()

    def update_file(self,
                    step,
                    epoch,
                    batch,
                    loss,
                    file_name):

        data_temp = {
            'Step': [int(step)],
            'Epoch_Number': [int(epoch)],
            'Batch_number': [int(batch)],
            'Loss': [float(loss)]
        }

        with open(self.path + '/' + file_name, 'r') as file:
            data = (json.load(file))

        for key, value in zip(data.items(), data_temp.items()):
            key[1].append(value[1][0])

        with open(self.path + '/' + file_name, 'w') as file:
            json.dump(data, file)

    def read_files(self,
                   file_name):

        with open(self.path + '/' + file_name, 'r') as file:
            data = (json.load(file))

        dataframe = []
        for column in data.keys():
            dataframe.append(pd.DataFrame(data[column], columns=[column]))

        return pd.concat(dataframe, axis=1)

    def start(self,
              name=None):

        if name is not None:

            self.path = self.file_path + '/' + name

            if not os.path.exists(self.path):
                os.makedirs(self.path)
                self.open_writers()
            else:
                print('removing')
                shutil.rmtree(self.path)
                os.makedirs(self.path)
                self.open_writers()
        else:
            self.open_writers()

    def write_results(self,
                      predictions,
                      labels,
                      dataframe,
                      mse,
                      mae):

        np.save(self.path + '/predictions.npy', predictions)
        np.save(self.path + '/labels.npy', labels)

        dataframe.to_csv(self.path + '/results.csv')

        with open(self.path + '/final_results.txt', 'w') as file:
            file.write('Mean Squared Error - {}\n'.format(mse))
            file.write('Mean Absolute Error - {}'.format(mae))

'''
cell5
'''
import os, time

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# import hdbscan

SEED = 1337

sensor_columns = ["sensor {}".format(s) for s in range(1,22)]

info_columns = ['dataset_id', 'unit_id','cycle','setting 1', 'setting 2', 'setting 3']

label_columns = ['dataset_id', 'unit_id', 'rul']

settings = ['setting 1', 'setting 2', 'setting 3']
type_1 = ['FD001', 'FD003']
type_2 = ['FD002', 'FD004']

class DataReader(object):
    def __init__(self,
                 raw_data_path_train,
                 raw_data_path_test,
                 **kwargs):

        assert os.path.isfile(raw_data_path_train) is True, \
            'This file do not exist. Please select an existing file'

        assert os.path.isfile(raw_data_path_test) is True, \
            'This file do not exist. Please select an existing file'

        assert raw_data_path_train.lower().endswith(('.csv', '.parquet', '.hdf5', '.pickle', '.pkl')) is True, \
            'This class can\'t handle this extension. Please specify a .csv, .parquet, .hdf5, .pickle extension'

        assert raw_data_path_test.lower().endswith(('.csv', '.parquet', '.hdf5', '.pickle', 'pkl')) is True, \
            'This class can\'t handle this extension. Please specify a .csv, .parquet, .hdf5, .pickle extension'

        self.raw_data_path_train = raw_data_path_train
        self.raw_data_path_test = raw_data_path_test
        self.loader_engine(**kwargs)
        self.train = self.loader_train()
        self.test = self.loader_test()

    def loader_engine(self, **kwargs):
        if self.raw_data_path_train.lower().endswith(('.csv')):
            self.loader_train = lambda: pd.read_csv(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_csv(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.parquet')):
            self.loader_train = lambda: pd.read_parquet(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_parquet(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.hdf5')):
            self.loader_train = lambda: pd.read_hdf(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_hdf(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.pkl', 'pickle')):
            self.loader_train = lambda: pd.read_pickle(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_pickle(self.raw_data_path_test, **kwargs)

    def calculate_unique_turbines(self):

        self.train_turbines = np.arange(len(self.train.index.to_series().unique()))
        self.test_turbines = np.arange(len(self.test.index.to_series().unique()))

    def cluestering(self, train, validation, test=None, min_cluster_size=100):

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True).fit(
            train[['setting 1', 'setting 2', 'setting 3']])

        train_labels, strengths = hdbscan.approximate_predict(clusterer, train[['setting 1', 'setting 2', 'setting 3']])
        validation_labels, strengths = hdbscan.approximate_predict(clusterer,
                                                                   validation[['setting 1', 'setting 2', 'setting 3']])

        train['HDBScan'] = train_labels
        validation['HDBScan'] = validation_labels

        if test is not None:
            test_labels, strengths = hdbscan.approximate_predict(clusterer,
                                                                 test[['setting 1', 'setting 2', 'setting 3']])
            test['HDBScan'] = test_labels

            return train, validation, test
        else:
            return train, validation

    def normalize_by_type(self, train, validation, normalization, test=None):

        df_train_type1 = train.loc[type_1]
        df_train_type2 = train.loc[type_2]
        
        df_validation_type1 = validation.loc[type_1]
        df_validation_type2 = validation.loc[type_2]
        
        if len(df_train_type1):
            
            df_train_type1_normalize = df_train_type1.copy()
            df_validation_type1_normalize = df_validation_type1.copy()

            if normalization == 'Standardization':
                scaler_type1 = StandardScaler().fit(df_train_type1[sensor_columns])
            elif normalization == 'MinMaxScaler':
                scaler_type1 = MinMaxScaler().fit(df_train_type1[sensor_columns])

            df_train_type1_normalize[sensor_columns] = scaler_type1.transform(df_train_type1[sensor_columns])
            df_validation_type1_normalize[sensor_columns] = scaler_type1.transform(df_validation_type1[sensor_columns])

            df_train_type1 = df_train_type1_normalize.copy()
            df_validation_type1 = df_validation_type1_normalize.copy()

            del (df_train_type1_normalize, df_validation_type1_normalize)

        df_train_type2_normalize = df_train_type2.copy()
        df_validation_type2_normalize = df_validation_type2.copy()

        gb = df_train_type2.groupby('HDBScan')[sensor_columns]
        
        d = {}

        for x in gb.groups:
            if normalization == 'Standardization':
                d["scaler_type2_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
            elif normalization == 'MinMaxScaler':
                d["scaler_type2_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))

            df_train_type2_normalize.loc[df_train_type2_normalize['HDBScan'] == x, sensor_columns] = d[
                "scaler_type2_{0}".format(x)].transform(
                df_train_type2.loc[df_train_type2['HDBScan'] == x, sensor_columns])
            df_validation_type2_normalize.loc[df_validation_type2_normalize['HDBScan'] == x, sensor_columns] = d[
                "scaler_type2_{0}".format(x)].transform(
                df_validation_type2.loc[df_validation_type2['HDBScan'] == x, sensor_columns])

        df_train_type2 = df_train_type2_normalize.copy()
        df_validation_type2 = df_validation_type2_normalize.copy()

        del (df_train_type2_normalize, df_validation_type2_normalize)

        df_train_all = pd.concat([df_train_type1, df_train_type2])
        df_validation_all = pd.concat([df_validation_type1, df_validation_type2])

        if test is not None:
            df_test_type1 = test.loc[type_1]
            df_test_type2 = test.loc[type_2]
            
            if len(df_test_type1):
                
                df_test_type1_normalize = df_test_type1.copy()
                df_test_type1_normalize[sensor_columns] = scaler_type1.transform(df_test_type1[sensor_columns])
                df_test_type1 = df_test_type1_normalize.copy()

                del(df_test_type1_normalize)

            df_test_type2_normalize = df_test_type2.copy()

            for x in gb.groups:
                df_test_type2_normalize.loc[df_test_type2_normalize['HDBScan'] == x, sensor_columns] = d[
                    "scaler_type2_{0}".format(x)].transform(
                    df_test_type2.loc[df_test_type2['HDBScan'] == x, sensor_columns])

            df_test_type2 = df_test_type2_normalize.copy()

            del(df_test_type2_normalize)

            df_test_all = pd.concat([df_test_type1, df_test_type2])

            return df_train_all, df_validation_all, df_test_all
        else:
            return df_train_all, df_validation_all
    
    def binarize(self, train, validation, test=None):
        
#         setting_operational = ["setting_op {}".format(s) for s in range(1, 7)]
#         dataset_id_columns = ["dataset_id {}".format(s) for s in range(1, 5)]
        n = len(train.groupby('HDBScan'))
        setting_operational = ["setting_op {}".format(s) for s in range(1, n+1)]
        n = len(train.groupby('dataset_id'))
        dataset_id_columns = ["dataset_id {}".format(s) for s in range(1, n+1)]
        
        preprocess_HDBscan = LabelBinarizer()
        preprocess_ID = LabelBinarizer()

        preprocess_HDBscan.fit(train['HDBScan'])
        preprocess_ID.fit(train.reset_index()['dataset_id'])

        dataframe_HDBscan = pd.DataFrame(preprocess_HDBscan.transform(train['HDBScan']),
                                         columns=setting_operational)
        dataframe_dataset_id = pd.DataFrame(preprocess_ID.transform(train.reset_index()['dataset_id']),
                                            columns=dataset_id_columns)

        dataframe_HDBscan_validation = pd.DataFrame(preprocess_HDBscan.transform(validation['HDBScan']),
                                                    columns=setting_operational)
        dataframe_dataset_id_validation = pd.DataFrame(
            preprocess_ID.transform(validation.reset_index()['dataset_id']), columns=dataset_id_columns)

        train = train.reset_index().join(dataframe_HDBscan)
        train = train.join(dataframe_dataset_id)

        validation = validation.reset_index().join(dataframe_HDBscan_validation)
        validation = validation.join(dataframe_dataset_id_validation)

        if test is not None:
            dataframe_HDBscan_test = pd.DataFrame(preprocess_HDBscan.transform(test['HDBScan']),
                                                  columns=setting_operational)
            dataframe_dataset_id_test = pd.DataFrame(preprocess_ID.transform(test.reset_index()['dataset_id']),
                                                     columns=dataset_id_columns)

            test = test.reset_index().join(dataframe_HDBscan_test)
            test = test.join(dataframe_dataset_id_test)

            return train, validation, test
        else:
            return train, validation

    def transform_data(self, df, length_sequence):

        array_data = []
        array_data_label = []

        for index_train in (df.index.to_series().unique()):
            temp_df_train = df.loc[index_train]

            for i in range(1, len(temp_df_train)+1):
                train_x = np.ones((length_sequence, (len(temp_df_train.columns) - 1))) * -1000
                train_y = np.ones((length_sequence, 1)) * -1000

                if i - length_sequence < 0:
                    x = 0
                else:
                    x = i - length_sequence

                data = temp_df_train.iloc[x:i]

                label = data['RUL'].copy().values
                data = data.drop(['RUL'], axis=1).values
                train_x[-len(data):, :] = data

                train_y[-len(data):, 0] = label
                array_data.append(train_x)
                array_data_label.append(train_y)

        return np.array(array_data), np.array(array_data_label)

    def prepare_datareader(self, batch_size, validation_split, number_steps_train, normalization):

        train_turbines, validation_turbines = train_test_split(self.train_turbines, test_size=validation_split)

        idx_train = self.train.index.to_series().unique()[train_turbines]
        idx_validation = self.train.index.to_series().unique()[validation_turbines]
        idx_test = self.test.index.to_series().unique()[self.test_turbines]

        train = self.train.loc[idx_train]
        validation = self.train.loc[idx_validation]
        test = self.test.loc[idx_test]

        train, validation, test = self.normalize_by_type(train, validation, normalization, test)
        
        train, validation, test = self.binarize(train, validation, test)
        

        train = train.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 6','sensor 10', 'sensor 16',
             'sensor 18', 'sensor 19','delta','dataset_id 1','local_density'], axis=1)
        validation = validation.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5',  'sensor 6','sensor 10', 'sensor 16',
             'sensor 18', 'sensor 19','delta','dataset_id 1','local_density'], axis=1)
        test = test.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5',  'sensor 6', 'sensor 10', 'sensor 16',
             'sensor 18', 'sensor 19','delta','dataset_id 1','local_density'], axis=1)
        
        
        # show the full columns and values 
        pd.set_option('display.max_columns',1000)
        pd.set_option('display.width',1000)
        pd.set_option('display.max_colwidth',1000)
        print('features used in this term', train.columns)
        print('values used in this term', train.head(4))
        
     #   print(dataset_name)
        
        if dataset_name == 'FD001' or dataset_name == 'FD003':
            
            train = train.drop(['setting_op 1'],axis = 1)
            validation = validation.drop(['setting_op 1'],axis = 1)
            test = test.drop(['setting_op 1'],axis = 1)
#        print(train.head(3))
            
        # print('train colums',train.shape,train.columns)
        
        doc=open('D:/RUL/results/%s/2.txt'
             %(dataset_name),'w')   
        
    
        # if the file does not exists, system will creat this file automatically; 
        # 'a'represent we could write in the content constantly and remain the old content.
        # there are many modes for this write-in scheme:（'w+','w','wb'）
        # doc.write(dataset_name) 
        # using '.write()' or using 'print()' are both okay
        print('columns used', train.columns,file=doc)
        
        doc.close()
        print("columns used have been saved")

        self.train_data, self.train_label_data = self.transform_data(train, number_steps_train)
        self.validation_data, self.validation_label_data  = self.transform_data(validation, number_steps_train)
        self.test_data, self.test_label_data = self.transform_data(test, number_steps_train)

        self.train_length = len(self.train_data)
        self.validation_length = len(self.validation_data)
        self.test_length = len(self.test_data)

        self.train_steps = round(len(self.train_data) / batch_size + 0.5)
        self.validation_steps = round(len(self.validation_data) / batch_size + 0.5)
        self.test_steps = round(len(self.test_data) / batch_size + 0.5)

        self.train_generator = self.generator_train(batch_size)
        self.validation_generator = self.generator_validation(batch_size)
        self.test_generator = self.generator_test(batch_size)

    def calculate_unique_turbines_cv(self, splits=5):

        cv = []
        cv_val = []

        split = KFold(n_splits=splits, shuffle=True)

        for train_index, validation_index in split.split(np.arange(len(self.train.index.to_series().unique()))):
            cv.append(train_index)
            cv_val.append(validation_index)

        return cv, cv_val

    def prepare_datareader_cv(self, splits, batch_size, number_steps_train):

        cv_indexes, cv_val_indexes = self.calculate_unique_turbines_cv(splits)

        idx_train = self.train.index.to_series().unique()[cv_indexes]
        idx_val = self.train.index.to_series().unique()[cv_val_indexes]

        train = self.train.loc[idx_train]
        validation = self.train.loc[idx_val]

        train, validation = self.normalize_by_type(train, validation)

        train, validation = self.binarize(train, validation)

        train = train.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 6','sensor 10', 'sensor 16', 'sensor 18', 'sensor 19','local_density'], axis=1)
        validation = validation.set_index(['dataset_id', 'unit_id']).drop(
            ['HDBScan', 'cycle', 'setting 1', 'setting 2', 'setting 3',
             'sensor 1', 'sensor 5', 'sensor 6','sensor 10', 'sensor 16', 'sensor 18', 'sensor 19','local_density'], axis=1)
        
         
        
        # 不省略，完全显示
        pd.set_option('display.max_columns',1000)
        pd.set_option('display.width',1000)
        pd.set_option('display.max_colwidth',1000)
        print('features used in this term (cv)', train.columns)
        print('values used in this term (cv)', train.head(4))
        
        self.train_data, self.train_label_data = self.transform_data(train, number_steps_train)
        self.validation_data, self.validation_label_data = self.transform_data(validation, number_steps_train)

        self.train_length = len(self.train_data)
        self.validation_length = len(self.validation_data)

        self.train_steps = round(len(self.train_data) / batch_size + 0.5)
        self.validation_steps = round(len(self.validation_data) / batch_size + 0.5)

        self.train_generator = self.generator_train(batch_size)
        self.validation_generator = self.generator_validation(batch_size)

    def generator_train(self, batch_size):
        while True:
            self.train_data, self.train_label_data = shuffle(self.train_data, self.train_label_data, random_state=1337)
            for ndx in range(0, self.train_length, batch_size):
                yield self.train_data[ndx:min(ndx + batch_size, self.train_length)], self.train_label_data[ndx:min(ndx + batch_size, self.train_length)]

    def generator_validation(self, batch_size):
        while True:
            for ndx in range(0, self.validation_length, batch_size):
                yield self.validation_data[ndx:min(ndx + batch_size, self.validation_length)], self.validation_label_data[
                                                                      ndx:min(ndx + batch_size, self.validation_length)]

    def generator_test(self, batch_size):
        while True:
            for ndx in range(0, self.test_length, batch_size):
                yield self.test_data[ndx:min(ndx + batch_size, self.test_length)], self.test_label_data[ndx:min(ndx + batch_size, self.test_length)]
                
'''
cell6
'''
def check_data():
    train_path = './datadrive/Turbofan_Engine/FD002/df_train_cluster_piecewise.pkl'
    test_path = './datadrive/Turbofan_Engine/FD002/df_test_cluster_piecewise.pkl'
    datareader_test = DataReader(train_path, test_path)
    datareader_test.calculate_unique_turbines()
    datareader_test.prepare_datareader(128, 0.2, 100, 'Standardization')
    print(datareader_test.train_data.shape)

# check_data()
    
'''
cell7
'''
for paopaopao in range(1):
    
    print('RUN NO.', paopaopao+1)
    

    dropout_train = 0.1
    
    
    # num_input, dataset_name, total_steps = 14, 'FD001', 6000
    num_input, dataset_name, total_steps = 20, 'FD002', 6000
#    num_input, dataset_name, total_steps = 14, 'FD003', 6000
    # num_input, dataset_name, total_steps = 20, 'FD004', 12000
    
   
    loss_func = 'MSE_SCORE'  # 'SCORE','MSE','MSE_SCORE'
    
    model = RNNTrainer(train_path = 'D:/RUL/data_xiaozhi_xixi/%s/train_xiaozhi_xixi.pkl'%dataset_name,
                       test_path = 'D:/RUL/data_xiaozhi_xixi/%s/test_xiaozhi_xixi.pkl'%dataset_name,
                       logger_path = 'D:/RUL/logs/temp_logger/',
                       model_name = 'Turbofan_Test',
                       train_log_interval = 100,
                       valid_log_interval = 100,
                       validation_split = 0.2,
                       use_script=True,
                       lr = 1e-4,   #learning rate  default value 1e-4
                       max_lr = (1e-3)*8,   #max learning rate  default 1e-2
                       total_steps = total_steps,
                       number_steps_train = 15,   #window_size
                       hidden_size = 256,
                       num_layers = 2,
                       cell_type = 'GRU',
                       kernel_size=10,
                       batch_size = 256,
                       num_epoch = 15,
    #                    number_features_input = 17,
                       number_features_input = num_input,
                       number_features_output = 17,
                       # loss_function = 'MSE',
                       #loss_function = 'SCORE',
                       loss_function = loss_func,  #define the loss outside in order to recorde the output
                       optimizer = 'Adam',
                       normalizer = 'Standardization',
    #                    normalizer = 'MinMaxScaler',
    #                    use_scheduler = False,
    #                    use_scheduler = 2,
                       use_scheduler = 3,     #using clyclic learning rate
                       use_cuda = True,
                      )
    
    '''cell7
    '''
    warnings.filterwarnings("ignore")
    print('use_cuda', model.use_cuda)
    
    model.train(10)
    
    '''cell8
    '''
    print(model.filelogger.path)
    model.get_best('load')
    
    predictions, labels = model.predict()
    print(predictions.shape, labels.shape)
    df_test, results, mse, mae, r2, score = model.postprocess(predictions, labels)
    
    # save the best one
    shutil.copyfile(model.get_best('save'),'D:/RUL/results/best/xixi/%s/%.3f.pth'%(dataset_name,score))
    
    print()
    print(dataset_name)
    print('='*36)
    print('Score:\t\t%.3f'% score)
    print('Accuracy:\t%.3f'% (results['Accuracy'].mean()*100) )
    print('RMSE:\t\t%.3f'% np.sqrt(mse))
    print('mse:\t\t%.3f'% mse)
    print('mae:\t\t%.3f'% mae)
    print('score_avg:\t%.3f'% results['Score'].mean())
    print('R2:\t\t%.3f'% r2)
    print('='*36)
    

    
    #for showing and recording the results
    
    doc=open('D:/RUL/results/%s/2.txt'
             %(dataset_name),'a')   
    
    # if the file does not exists, system will creat this file automatically; 
    # 'a'represent we could write in the content constantly and remain the old content.
    # there are many modes for this write-in scheme:（'w+','w','wb'）
    # doc.write(dataset_name) 
    # using '.write()' or using 'print()' are both okay
    print(dataset_name,file=doc)
    print('='*36,file=doc)
    print('Score:\t\t%.3f'% score,file=doc)
    print('Accuracy:\t%.3f'% (results['Accuracy'].mean()*100),file=doc )
    print('RMSE:\t\t%.3f'% np.sqrt(mse),file=doc)
    print('mse:\t\t%.3f'% mse,file=doc)
    print('mae:\t\t%.3f'% mae,file=doc)
    print('score_avg:\t%.3f'% results['Score'].mean(),file=doc)
    print('R2:\t\t%.3f'% r2,file=doc)
    print('='*36,file=doc)
    print('loss function', loss_func,file=doc)
    
    doc.close( )
    
    # alter the file name
    os.rename('D:/RUL/results/%s/2.txt'
             %(dataset_name), 'D:/RUL/results/%s/score %.3f RMSE %.4f.txt'
             %(dataset_name,score,np.sqrt(mse)),)
    
    print("perfoemace results have been saved")
    
    
    # the length of a and b should be consistent (the same)
    # the key value in the dictionary is corresponding to the name of the column in csv file 
    dataframe = pd.DataFrame({'predictede RUL':results['Predicted_RUL'],'real_RUL':results['True_RUL']})
    
    # transfer the DataFrame data into csv files,using the index to decide wheather to show the name of columns, default=True
    dataframe.to_csv("D:/RUL/results/%s./%.4f_%.4f.csv"%(dataset_name,score,np.sqrt(mse)),sep=',')
    df_test.to_csv('D:/RUL/results/{}/score{}_RMSE{}df_test.csv'.format(dataset_name,score,np.sqrt(mse)))

    print("RUL results have been saved")
    
    #'''
    #cell12
    #'''
    #a,b = model.predict()
    #
    #'''
    #cell13
    #'''
    #def module(x):
    #    if x<0:
    #        return 0
    #    else:
    #        return x
    #
    #c = pd.DataFrame(a[:,-1,0])[0].apply(lambda x: module(x))
    #
    #'''
    #cell14
    #'''
    #pd.DataFrame(model.datareader.test_data[:,-1,:])
