# -*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import numpy as np


class Preprocessing(object):
    
    def __init__(self,
                 dataset_name,
                 **kwargs):     
        
        self.dataset_name = dataset_name
        self._loader_engine(**kwargs)
        self._add_columns_name()
        self._RUL_labeling()
        #self._normal()
        self._Clustering(cluster='Dbscan')
             
        # data load
    def _loader_engine(self, **kwargs):
        self.train_data = pd.read_csv( './data/{}/train_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs)
        self.test_data = pd.read_csv( './data/{}/test_{}.csv'.format(self.dataset_name,self.dataset_name) ,header=None, **kwargs)
        self.data_test_RUL =  pd.read_csv('./data/{}/RUL_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs)
        

    def _add_columns_name(self):
        sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
        info_columns = ['unit_id','cycle']
        settings_columns = ['setting 1', 'setting 2', 'setting 3']
        self.train_data.columns = info_columns + settings_columns + sensor_columns
        self.test_data.columns = info_columns + settings_columns + sensor_columns
        self.data_test_RUL.columns = ['RUL']
        self.data_test_RUL['unit_id'] = [i for i in range(1,len(self.data_test_RUL)+1,1)]
        self.data_test_RUL.set_index('unit_id',inplace=True,drop=True)
        
        self.train_data['dataset_id'] = [self.dataset_name]*len(self.train_data['unit_id'])
        self.test_data['dataset_id'] = [self.dataset_name]*len(self.test_data['unit_id'])
            

    def _RUL_labeling(self):
        #train
        maxRUL_dict = self.train_data.groupby('unit_id')['cycle'].max().to_dict()
        self.train_data['maxRUL'] = self.train_data['unit_id'].map(maxRUL_dict)
        self.train_data['RUL'] = self.train_data['maxRUL'] - self.train_data['cycle']
        self.train_data.drop(['maxRUL'],inplace=True,axis=1)
        filterRUL = (self.train_data['RUL']>130)
        self.train_data.loc[filterRUL,['RUL']] = 130
       
        #test
        RUL_dict = self.data_test_RUL.to_dict()
        self.test_data['RUL_test'] =self.test_data['unit_id'].map(RUL_dict['RUL'])
        maxT_dict_train = self.test_data.groupby('unit_id')['cycle'].max().to_dict()
        self.test_data['maxT'] = self.test_data['unit_id'].map(maxT_dict_train)
        self.test_data['RUL'] = self.test_data['RUL_test'] + self.test_data['maxT'] - self.test_data['cycle']
        filterRUL_test = (self.test_data['RUL']>130)
        self.test_data.loc[filterRUL_test,['RUL']] = 130
        self.test_data.drop(['RUL_test','maxT'],inplace=True,axis=1)

        return self.train_data, self.test_data
    
    def _normal(self):        
        Scaler = StandardScaler().fit(self.train_data[['setting 1', 'setting 2', 'setting 3']])
        self.train_data[['setting 1', 'setting 2', 'setting 3']] = Scaler.transform(self.train_data[['setting 1', 'setting 2', 'setting 3']])
        self.test_data[['setting 1', 'setting 2', 'setting 3']] = Scaler.transform(self.test_data[['setting 1', 'setting 2', 'setting 3']])
            
        
    def _Clustering(self,cluster): 
        
        if cluster == 'Hdbscan':  
            import hdbscan
            clustering = hdbscan.HDBSCAN(min_cluster_size=2000, prediction_data=True)
            clustering.fit(self.train_data[['setting 1', 'setting 2', 'setting 3']])
            train_labels = clustering.labels_
            test_labels, strengths = hdbscan.approximate_predict(clustering, self.test_data[['setting 1', 'setting 2', 'setting 3']])
            
        elif cluster == 'Dbscan': 
            clustering = DBSCAN(eps=3, min_samples=2000).fit(self.train_data[['setting 1', 'setting 2', 'setting 3']])
            train_labels = clustering.labels_
            self.train_data['HDBScan'] = train_labels      
            test_labels = clustering.fit_predict(self.test_data[['setting 1', 'setting 2', 'setting 3']])
                        

        self.train_data['HDBScan'] = train_labels
        self.test_data['HDBScan'] = test_labels
        
        
        self.train_data.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
        self.test_data.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
        
        self.train_data.to_csv('./data/{}/train_op.csv'.format(self.dataset_name))
        self.test_data.to_csv('./data/{}/test_op.csv'.format(self.dataset_name))

#preprocess = Preprocessing(dataset_name='FD002')
    




