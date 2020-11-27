# -*- coding: utf-8 -*-
"""
cell_7: main
"""

import random

SEED = 1337

sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
features = sensor_columns + ['local_density']

info_columns = ['dataset_id', 'unit_id','cycle','setting 1', 'setting 2', 'setting 3']

label_columns = ['dataset_id', 'unit_id', 'rul']

settings = ['setting 1', 'setting 2', 'setting 3']
type_1 = ['FD001', 'FD003']
type_2 = ['FD002', 'FD004']


def check_data():
    train_path = './datadrive/Turbofan_Engine/FD002/df_train_cluster_piecewise.pkl'
    test_path = './datadrive/Turbofan_Engine/FD002/df_test_cluster_piecewise.pkl'
    datareader_test = DataReader(train_path, test_path)
    datareader_test.calculate_unique_turbines()
    datareader_test.prepare_datareader(128, 0.2, 100, 'Standardization')
    print(datareader_test.train_data.shape)

# check_data()
    

run_sum = 1
for paopaopao in range(run_sum):
    
    print('RUN NO.', paopaopao+1, 'in', run_sum)
    

    dropout_train = 0.1
    
#without density    
    # num_input, dataset_name, total_steps = 14, 'FD001', 6000
    num_input, dataset_name, total_steps = 20, 'FD002', 6000
#    num_input, dataset_name, total_steps = 14, 'FD003', 6000
    # num_input, dataset_name, total_steps = 20, 'FD004', 12000

#with density    
    # num_input, dataset_name, total_steps = 15, 'FD001', 6000
    num_input, dataset_name, total_steps = 21, 'FD002', 6000
#    num_input, dataset_name, total_steps = 15, 'FD003', 6000
    # num_input, dataset_name, total_steps = 21, 'FD004', 12000
   
    loss_func = 'MSE_SCORE'  # 'SCORE','MSE','MSE_SCORE'
    max_lr = (1e-3)*5.5
    # get a random number
    random_steps = random.randint(15,48)
    window_size = 15
    
    model = RNNTrainer(train_path = 'D:/RUL/data_xiaozhi_xixi/%s/train_xiaozhi_xixi.pkl'%dataset_name,
                       test_path = 'D:/RUL/data_xiaozhi_xixi/%s/test_xiaozhi_xixi.pkl'%dataset_name,
                       logger_path = 'D:/RUL/logs/temp_logger/',
                       model_name = 'Turbofan_Test',
                       train_log_interval = 100,
                       valid_log_interval = 100,
                       validation_split = 0.2,
                       use_script=True,
                       lr = 1e-4,   #learning rate  default value 1e-4
                       max_lr = max_lr,   #max learning rate  default 1e-2
                       total_steps = total_steps,
                       number_steps_train = window_size,   #window_size
                       hidden_size = 256,
                       num_layers = 2,
                       cell_type = 'GRU',
                       kernel_size=10,
                       batch_size = 256,
                       num_epoch =  random_steps,
    #                    number_features_input = 17,
                       number_features_input = num_input,
                       number_features_output = 15,
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
    

    warnings.filterwarnings("ignore")
    print('use_cuda', model.use_cuda)
    
    model.train(10)
    

    print(model.filelogger.path)
    model.get_best('load')
    
    predictions, labels = model.predict()
    print(predictions.shape, labels.shape)
    df_test, results, mse, mae, r2, score = model.postprocess(predictions, labels)
    
    # save the best one
    shutil.copyfile(model.get_best('save'),'D:/RUL/results_with_density/best/xixi/%s/%.3f.pth'%(dataset_name,score))
    
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
    print('max-lr', max_lr)
    print('loss function', loss_func)
    print('number_steps_train', random_steps)
    print('window size', window_size)
    print('='*36)
    

    
    #for showing and recording the results
    
    doc=open('D:/RUL/results_with_density/%s/2.txt'
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
    print('max-lr', max_lr,file=doc)
    print('number_steps_train', random_steps,file=doc)
    print('window size', window_size,file=doc)
    doc.close( )
    
    # alter the file name
    os.rename('D:/RUL/results_with_density/%s/2.txt'
             %(dataset_name), 'D:/RUL/results_with_density/%s/score %.3f RMSE %.4f.txt'
             %(dataset_name,score,np.sqrt(mse)),)
    
    print("perfoemace results have been saved")
    
    
    # the length of a and b should be consistent (the same)
    # the key value in the dictionary is corresponding to the name of the column in csv file 
    dataframe = pd.DataFrame({'predictede RUL':results['Predicted_RUL'],'real_RUL':results['True_RUL']})
    
    # transfer the DataFrame data into csv files,using the index to decide wheather to show the name of columns, default=True
    dataframe.to_csv("D:/RUL/results_with_density/%s./%.4f_%.4f.csv"%(dataset_name,score,np.sqrt(mse)),sep=',')
    df_test.to_csv('D:/RUL/results_with_density/{}/score{}_RMSE{}df_test.csv'.format(dataset_name,score,np.sqrt(mse)))

    print("RUL results have been saved")