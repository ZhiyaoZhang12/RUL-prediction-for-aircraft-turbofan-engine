# -*- coding: utf-8 -*-

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
from filelogger import FileLogger
from DataReader import DataReader


SEED = 1337

def error_function(df, y_predicted, y_true):
    return int(df[y_predicted] - df[y_true])


def cra_function(df, y_predicted, y_true):
    cra = 1 - abs(((df[y_predicted] - df[y_true])/df[y_true]))
    return cra 

def score_function(df, label, alpha1=13, alpha2=10):
    if df[label] <= 0:
        return (np.exp(-(df[label] / alpha1)) - 1)

    elif df[label] > 0:
        return (np.exp((df[label] / alpha2)) - 1)

def accuracy_function(df, label, alpha1=13, alpha2=10):
    if df[label]<-alpha1 or df[label]>alpha2:
        return 0
    return 1




class Trainer(object):
    def __init__(self,
                 dataset_name,
                 feature_used,
                 sensor_feature_used,
                 train_path,
                 test_path,
                 logger_path,
                 model_name,
                 train_log_interval,
                 valid_log_interval,
                 is_MOC_normal,
                 load_model_name=None,
                 use_script=False,
                 use_cuda = True,
                 **kwargs):

        # Data Reader
        self.datareader = DataReader(train_path,
                                     test_path,
                                     dataset_name,
                                     feature_used,
                                     sensor_feature_used,
                                     is_MOC_normal = is_MOC_normal,
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
        self.dataset_name = dataset_name

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

    def train(self, patience):

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
                validation_loss = round(validation_loss,4)
                print('\n','                         train_loss',train_loss,'val_loss:',validation_loss)

                self.tensorboard.add_scalar('Validation Mean Squared Error loss per epoch',
                                            validation_loss,
                                            epoch)

                if self.use_scheduler==1:
                    self.scheduler.step(validation_loss)
                elif self.use_scheduler>=2:
                    self.scheduler.step()

                if validation_loss < best_validation_loss:
                    previous_best_loss = best_validation_loss
                    best_validation_loss = validation_loss
                    best_validation_epoch = epoch + 1
                    patience_step = 0
                    self.save('Model_Checkpoint' + str(epoch + 1) + '_valid_loss_' + str(best_validation_loss) + '.pth')
                    
                    doc=open('./results/%s/2.txt'%(self.dataset_name),'a')
                    print('{} epoch, validation_loss'.format(epoch + 1), validation_loss,file=doc)
                    doc.close()
                    
                    ## ealy end (add in 2021.0817)
                    if epoch >= 1:
                        if best_validation_loss/previous_best_loss < 0.1:
                            print('-------------------------- early stop! --------------------------')
                            doc=open('./results/%s/2.txt'%(self.dataset_name),'a')
                            print('-------------------------- early stop! --------------------------',file=doc)
                            doc.close()
                            return best_validation_loss
                            
                    
                    
                else:
                    patience_step += 1
                    if patience_step >= patience:
                        print('Train is donne, {} epochs in a row without improving validation loss!'.format(patience))
                        
                        doc=open('./results/%s/2.txt'
                             %(self.dataset_name),'a')
                        print('\n','best_validation_loss', best_validation_loss,file=doc)
                        doc.close()
                        
                        print('\n',"best_validation_loss have been saved")
        
                        return best_validation_loss

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
        df_test['CRA'] = df_test.apply(lambda df: cra_function(df, 'Predicted_RUL', 'True_RUL'), axis=1)


        df_temp = []
        for index in df_test.index.to_series().unique():
            df_temp.append(df_test.loc[index].iloc[-1][['True_RUL', 'Predicted_RUL', 'Error', 'Score','CRA']])

        results = pd.DataFrame(df_temp)

        mse = mean_squared_error(results['Predicted_RUL'], results['True_RUL'])
        mae = mean_absolute_error(results['Predicted_RUL'], results['True_RUL'])
        r2 = r2_score(results['True_RUL'], results['Predicted_RUL'])
        
        results['Accuracy'] = results.apply(lambda df: accuracy_function(df, 'Error'), axis=1)

        score = results['Score'].sum()
        cra_sum = results['CRA'].sum()
        cra = cra_sum/len(results['CRA'])

        return df_test, results, mse, mae, r2, score, cra


    def get_best(self,load_or_save):

        files = glob(self.filelogger.path + '/model_checkpoint/*')

        best = 10000000
        for file in files:
            number = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', file)
#             print(number)
            result = float(number[-1])
        
        
            if result < best:
                best = result
                
                # if load_or_save == 'load':
                #     doc=open('./results/%s/2.txt'
                #          %(self.dataset_name),'a')   

                #     print('validation_loss', result,file=doc)
                #     doc.close()
    #                print("validation_loss recored")
                    
                best_file = file

        if load_or_save == 'load':
            self.load(best_file)
        elif load_or_save == 'save':
            return(best_file)