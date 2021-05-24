# -*- coding: utf-8 -*-
          
import warnings
from RNN_model import RNNTrainer    

dropout_train = 0.5
Loss_function = 'SCORE'  #MSE, RMSE, SCORE, SCORE_MSE
window_size = 40

#     num_input, dataset_name, total_steps = 14, 'FD001', 6000
num_input, dataset_name, total_steps = 20, 'FD002', 6000
#    num_input, dataset_name, total_steps = 15, 'FD003', 6000
   # num_input, dataset_name, total_steps = 21, 'FD004', 12000

model = RNNTrainer(train_path = './data/%s/train_op.pkl'%dataset_name,
                   test_path = './data/%s/test_op.pkl'%dataset_name,
                   logger_path = './logs/',
                   model_name = 'Turbofan_Test',
                   dataset_name = dataset_name,
                   train_log_interval = 100,
                   valid_log_interval = 100,
                   validation_split = 0.2,
                   use_script=True,
                   lr = 1e-4,
                   max_lr = 1e-2,
                   total_steps = total_steps,
                   number_steps_train = window_size, 
                   hidden_size = 256,
                   num_layers = 2,
                   cell_type = 'GRU', # neural network strature selected
                   dropout_train = dropout_train,
                   kernel_size=10,
                   batch_size = 256,
                   num_epoch = 200,    # epochs for training
#                    number_features_input = 17,
                   number_features_input = num_input,
                   number_features_output = 1,
                   loss_function = Loss_function, 
                   optimizer = 'Adam',
                   normalizer = 'Standardization',
#                    normalizer = 'MinMaxScaler',
#                    use_scheduler = False,
#                    use_scheduler = 2,
                   use_scheduler = 3,     # use_scheduler = 3 : using cycling learning rate
                   use_cuda = True,
                  )



warnings.filterwarnings("ignore")
print('use_cuda:', model.use_cuda)

model.train(10)



import os, json, shutil
import numpy as np
import pandas as pd

print(model.filelogger.path)
model.get_best('load')

predictions, labels = model.predict()
print(predictions.shape, labels.shape)
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
     '='*36)
  
  
# save performance in txt file 
doc=open('./results/%s/2.txt'
         %(dataset_name),'a')   
   
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
print('window_size:', window_size,file=doc)
print('loss function', Loss_function,file=doc)
print('='*36,file=doc)

doc.close( )
os.rename('./results/%s/2.txt'
         %(dataset_name), './results/%s/score %.3f RMSE %.4f.txt'
         %(dataset_name,score,np.sqrt(mse)),)
print("perfoemace results have been saved")


# save results : predicted RUL and Real RUL
dataframe = pd.DataFrame({'predictede RUL':results['Predicted_RUL'],'real_RUL':results['True_RUL']})

# save the reults about 'predicted RUL' and 'real RUL'
dataframe.to_csv("./results/%s./%.4f_%.4f.csv"%(dataset_name,score,np.sqrt(mse)),sep=',')
df_test.to_csv('./results/{}/score{}_RMSE{}df_test.csv'.format(dataset_name,score,np.sqrt(mse)))

print("RUL results have been saved")
   
    
   
