# -*- coding: utf-8 -*-
          
import warnings
from RNN_model import RNNTrainer
import random


n = 10
for i in range(n):
    
    print('\n')
    print('This is the {} run'.format(i+1))
    window_size, is_one_hot,is_op_factor, is_MOC_normal= 40,1,1,1
    Loss_function = 'SCORE'

    dropout_train = 0.5     ### default = 0,0.5
    NN_type = 'GRU'  #LSTM, GRU, RNN
    num_layers = 2  #2
    sensor_feature_used = ['sensor 2', 'sensor 3', 'sensor 4', 'sensor 7', 'sensor 8', 'sensor 9',
        'sensor 11', 'sensor 12', 'sensor 13', 'sensor 14', 'sensor 15',
        'sensor 17', 'sensor 20', 'sensor 21']
    feature_used = sensor_feature_used + ['RUL']


    dataset_name, total_steps =  'FD001', 6000
    # dataset_name, total_steps = 'FD002', 6000
    #dataset_name, total_steps =  'FD003', 6000
    # dataset_name, total_steps =  'FD004', 12000
    

    if is_op_factor:
        feature_used += ['op factor']
    if  is_one_hot:
        feature_used += ["setting_op {}".format(s) for s in range(1,7)]

    num_input = len(feature_used) - 1     #minus RUL
    patience = 3 #default = 3
    
    
    model = RNNTrainer(train_path = './data/%s/train_op.csv'%dataset_name,   #train_op_normal
                       test_path = './data/%s/test_op.csv'%dataset_name,    #tets_op
                       logger_path = './logs/',
                       model_name = 'Turbofan_Test',
                       dataset_name = dataset_name,
                       train_log_interval = 100,
                       valid_log_interval = 100,
                       validation_split = 0.3,
                       use_script=True,
                       lr = 1e-4, #default = 4
                       max_lr = 1e-2, #default = 2
                       total_steps = total_steps,
                       number_steps_train = window_size, 
                       hidden_size = 256,
                       num_layers = num_layers,
                       cell_type = NN_type, # neural network strature selected default = 'GRU'
                       feature_used = feature_used,
                       sensor_feature_used = sensor_feature_used,
                       dropout_train = dropout_train,
                       kernel_size=10,
                       batch_size = 256,  #default=256
                       num_epoch = 100, #int(random.uniform(3, 30)),    # epochs for training   #default = 100
                       number_features_input = num_input,
                       number_features_output = 1,
                       loss_function = Loss_function, 
                       optimizer = 'Adam',
                       normalizer = 'Standardization',
                       use_scheduler = 3,     # use_scheduler = 3 : using cycling learning rate
                       use_cuda = True,
                       is_MOC_normal = is_MOC_normal
                      )
    
    
    
    warnings.filterwarnings("ignore")
    print('use_cuda:', model.use_cuda)
    
    model.train(patience)  #default patience = 3


    
    import os, json, shutil
    import numpy as np
    import pandas as pd
    
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
         'feature used',feature_used,'\n',
         '='*36)
      
      
    # save performance in txt file 
    doc=open('./results/%s/2.txt'
             %(dataset_name),'a')   
    print('\n',file = doc)
    print(dataset_name,file=doc)
    print('='*36,file=doc)
    print('Score:\t\t%.4f'% score,file=doc)
    print('Accuracy:\t%.4f'% (results['Accuracy'].mean()*100),file=doc )
    print('RMSE:\t\t%.4f'% np.sqrt(mse),file=doc)
    print('mse:\t\t%.4f'% mse,file=doc)
    print('mae:\t\t%.4f'% mae,file=doc)
    print('score_avg:\t%.4f'% results['Score'].mean(),file=doc)
    print('R2:\t\t%.4f'% r2,file=doc)
    print('CRA:\t\t%.4f'% cra,file=doc)
    print('dropout:\t\t%.1f'% dropout_train,file=doc)
    print('window_size:  ', window_size,file=doc)
    print('loss function:  ', Loss_function,file=doc)
    print('NN_type:  ', NN_type,file=doc)
    print('NN layers:  ',num_layers,file=doc)
    print('Num feature:  ',num_input,file=doc)
    print('patience:  ',patience,file=doc)
    print('is_MOC_normal:  ',is_MOC_normal,file=doc)
    print('feature used:  ',feature_used,file=doc)
    print('='*36,file=doc)

    doc.close( )
    
    if not os.path.exists('./results/%s/Score(%.3f)W(%s)OP_factor(%s)one-hot(%s)RMSE(%.3f).txt'
             %(dataset_name,score,window_size,is_op_factor,is_one_hot,np.sqrt(mse))):
        os.rename('./results/%s/2.txt'
                 %(dataset_name),'./results/%s/Score(%.3f)W(%s)OP_factor(%s)one-hot(%s)RMSE(%.3f).txt'
                 %(dataset_name,score,window_size,is_op_factor,is_one_hot,np.sqrt(mse)),)
        print("performace results have been saved!")
            # save results : predicted RUL and Real RUL
        dataframe = pd.DataFrame({'predictede RUL':results['Predicted_RUL'],'real_RUL':results['True_RUL']})
        
        # save the reults about 'predicted RUL' and 'real RUL'
        dataframe.to_csv("./results/%s/Test results_Score%.4fRMSE%.4f.csv"%(dataset_name,score,np.sqrt(mse)),sep=',')
        df_test.to_csv('./results/{}/All resutls_Score{:.4f}RMSE{:.4f}.csv'.format(dataset_name,score,np.sqrt(mse)))
        
        print("RUL results have been saved",'\n')
    else:
        print('similary results!')


