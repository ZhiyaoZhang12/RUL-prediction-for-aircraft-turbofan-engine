# -*- coding: utf-8 -*-
          
import warnings
from RNN_model import RNNTrainer
import random
import os, json, shutil
import numpy as np
import pandas as pd
from preprocessing import Preprocessing


n = 10
for i in range(n):
    
    print('\n')
    print('This is the {} run'.format(i+1))
    
    
    dataset_name, total_steps =  'FD001', 6000
    #dataset_name, total_steps = 'FD002', 6000
    #dataset_name, total_steps =  'FD003', 6000
    #dataset_name, total_steps =  'FD004', 12000
    
    
    window_size = 40   
    Loss_function = 'SCORE'  #score,mse,mae
    dropout_train = 0.5     
    NN_type = 'GRU'  #LSTM, GRU, RNN
    num_layers = 2  
    sensor_feature_used = ['sensor 2', 'sensor 3', 'sensor 4', 'sensor 7', 'sensor 8', 'sensor 9',
        'sensor 11', 'sensor 12', 'sensor 13', 'sensor 14', 'sensor 15',
        'sensor 17', 'sensor 20', 'sensor 21']
    columns_used = sensor_feature_used + ['RUL']   

    is_one_hot = True
    is_op_factor = True
    is_MOC_normal = True 
    
    if dataset_name=='FD001' or dataset_name == 'FD003':
        is_one_hot = False
  
    
    if is_op_factor:
        columns_used += ['op factor']
    if  is_one_hot:
        columns_used += ["setting_op {}".format(s) for s in range(1,7)]

    num_input = len(columns_used) - 1     #minus RUL
    patience = 3  #default = 3
    
    preprocess = Preprocessing(dataset_name)
    
    
    model = RNNTrainer(train_path = './data/%s/train_op.csv'%dataset_name,   
                       test_path = './data/%s/test_op.csv'%dataset_name,    
                       logger_path = './logs/',
                       model_name = 'Turbofan_Test',
                       dataset_name = dataset_name,
                       train_log_interval = 100,
                       valid_log_interval = 100,
                       validation_split = 0.3,
                       use_script=True,
                       lr = 1e-4, 
                       max_lr = 1e-2, 
                       total_steps = total_steps,
                       number_steps_train = window_size, 
                       hidden_size = 516,  #256
                       num_layers = num_layers,
                       cell_type = NN_type, 
                       columns_used = columns_used,
                       sensor_feature_used = sensor_feature_used,
                       dropout_train = dropout_train,
                       kernel_size=10,
                       batch_size = 256,  
                       num_epoch = 100,  
                       number_features_input = num_input,
                       number_features_output = 1,
                       loss_function = Loss_function, 
                       optimizer = 'Adam',
                       normalizer = 'Standardization',
                       use_scheduler = 3,     # using cycling learning rate
                       use_cuda = True,
                       is_MOC_normal = is_MOC_normal
                      )
    
    
    
    warnings.filterwarnings("ignore")
    print('use_cuda:', model.use_cuda)
    
    model.train(patience)  
    
    
    print(model.filelogger.path)
    model.get_best('load')
    
    predictions, labels = model.predict()
    # print(predictions.shape, labels.shape)
    df_test, results, mse, mae, r2, score, cra = model.postprocess(predictions, labels)
    
       
    # save the model with best results
    shutil.copyfile(model.get_best('save'),'./best/%s/%.3f.pth'%(dataset_name,score))
    print(
         dataset_name, '\n',
         '='*36, '\n',
         'Score:\t\t%.4f'% score, '\n',
         'Accuracy:\t%.4f'% (results['Accuracy'].mean()*100) , '\n',
         'RMSE:\t\t%.4f'% np.sqrt(mse), '\n',
         'mse:\t\t%.4f'% mse, '\n',
         'mae:\t\t%.4f'% mae, '\n',
         'score_avg:\t%.4f'% results['Score'].mean(), '\n',
         'R2:\t\t%.4f'% r2, '\n',
         'CRA:\t\t%.4f'% cra, '\n',
         'dropout:\t\t%.1f'% dropout_train, '\n',
         'window_size:', window_size, '\n',
         'loss function', Loss_function, '\n',
         'NN_type',NN_type,'\n',
         'NN layers',num_layers,'\n',
         'Num feature',num_input,'\n',
         'patience',patience,'\n',
         'is_MOC_normal',is_MOC_normal,'\n',
         'columns used',columns_used,'\n',
         '='*36)  

    dataframe = pd.DataFrame({'predictede RUL':results['Predicted_RUL'],'real_RUL':results['True_RUL']})        
    # save the reults about 'predicted RUL' and 'real RUL'
    dataframe.to_csv("./results/%s/Test results_Score%.4fRMSE%.4f.csv"%(dataset_name,score,np.sqrt(mse)),sep=',')
    print("RUL results have been saved",'\n')


