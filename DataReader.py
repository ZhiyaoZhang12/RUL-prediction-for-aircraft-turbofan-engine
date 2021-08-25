import os, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import hdbscan

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
                 dataset_name,
                 feature_used,
                 sensor_feature_used,
                 is_MOC_normal,
                 **kwargs):

        self.raw_data_path_train = raw_data_path_train
        self.raw_data_path_test = raw_data_path_test
        self.loader_engine(**kwargs)
        self.train = self.loader_train()
        self.test = self.loader_test()
        self.dataset_name = dataset_name
        self.feature_used = feature_used
        self.sensor_feature_used = sensor_feature_used
        self.is_MOC_normal = is_MOC_normal


    def loader_engine(self, **kwargs):
        if self.raw_data_path_train.lower().endswith(('.csv')):
            self.loader_train = lambda: pd.read_csv(self.raw_data_path_train, header = 0, index_col= ['dataset_id','unit_id'], **kwargs)
            self.loader_test = lambda: pd.read_csv(self.raw_data_path_test,header = 0, index_col= ['dataset_id','unit_id'], **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.parquet')):
            self.loader_train = lambda: pd.read_parquet(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_parquet(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.hdf5')):
            self.loader_train = lambda: pd.read_hdf(self.raw_data_path_train, **kwargs)
            self.loader_test = lambda: pd.read_hdf(self.raw_data_path_test, **kwargs)
        elif self.raw_data_path_train.lower().endswith(('.pkl', 'pickle')):
            self.loader_train = lambda: pd.read_pickle(self.raw_data_path_train)
            self.loader_test = lambda: pd.read_pickle(self.raw_data_path_test)


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
        
        if not self.is_MOC_normal:
            df_train_type1 = train
            df_validation_type1 = validation
            
            if len(df_train_type1):
                df_train_type1_normalize = df_train_type1.copy()
                df_validation_type1_normalize = df_validation_type1.copy()

            if normalization == 'Standardization':
                scaler_type1 = StandardScaler().fit(df_train_type1[self.sensor_feature_used])
            elif normalization == 'MinMaxScaler':
                scaler_type1 = MinMaxScaler().fit(df_train_type1[self.sensor_feature_used])

            df_train_type1_normalize[self.sensor_feature_used] = scaler_type1.transform(df_train_type1[self.sensor_feature_used])
            df_validation_type1_normalize[self.sensor_feature_used] = scaler_type1.transform(df_validation_type1[self.sensor_feature_used])

            df_train_type1 = df_train_type1_normalize.copy()
            df_validation_type1 = df_validation_type1_normalize.copy()

            del (df_train_type1_normalize, df_validation_type1_normalize)
            df_train = df_train_type1
            df_validation =  df_validation_type1
            
            if test is not None:
                df_test_type1 = test
                
                if len(df_test_type1):
                    df_test_type1_normalize = df_test_type1.copy()
                    df_test_type1_normalize[self.sensor_feature_used] = scaler_type1.transform(df_test_type1[self.sensor_feature_used])
                    df_test_type1 = df_test_type1_normalize.copy()
    
                    del(df_test_type1_normalize)
                    df_test = df_test_type1
                    return df_train, df_validation, df_test
            elif test is None:
                return df_train, df_validation


        if self.is_MOC_normal:

            df_train_type2 = train
            df_validation_type2 = validation
            
            df_train_type2_normalize = df_train_type2.copy()
            df_validation_type2_normalize = df_validation_type2.copy()
    
            gb = df_train_type2.groupby('HDBScan')[self.sensor_feature_used]
            
            d = {}
    
            for x in gb.groups:
                if normalization == 'Standardization':
                    d["scaler_type2_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
                elif normalization == 'MinMaxScaler':
                    d["scaler_type2_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
    
                df_train_type2_normalize.loc[df_train_type2_normalize['HDBScan'] == x, self.sensor_feature_used] = d[
                    "scaler_type2_{0}".format(x)].transform(
                    df_train_type2.loc[df_train_type2['HDBScan'] == x, self.sensor_feature_used])
                df_validation_type2_normalize.loc[df_validation_type2_normalize['HDBScan'] == x, self.sensor_feature_used] = d[
                    "scaler_type2_{0}".format(x)].transform(
                    df_validation_type2.loc[df_validation_type2['HDBScan'] == x, self.sensor_feature_used])
    
            df_train_type2 = df_train_type2_normalize.copy()
            df_validation_type2 = df_validation_type2_normalize.copy()
    
            del (df_train_type2_normalize, df_validation_type2_normalize)
    
            
            df_train = df_train_type2
            df_validation =  df_validation_type2


            if test is not None:
                df_test_type2 = test

                df_test_type2_normalize = df_test_type2.copy()
    
                for x in gb.groups:
                    df_test_type2_normalize.loc[df_test_type2_normalize['HDBScan'] == x, self.sensor_feature_used] = d[
                        "scaler_type2_{0}".format(x)].transform(
                        df_test_type2.loc[df_test_type2['HDBScan'] == x, self.sensor_feature_used])
    
                df_test_type2 = df_test_type2_normalize.copy()
                del(df_test_type2_normalize)
                
    
                df_test = df_test_type2
                return df_train, df_validation, df_test
            elif test is None:
                return df_train, df_validation
        

    ###one-hot coding
    def binarize(self, train, validation, test=None):
        
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
        
        train.set_index(['unit_id','dataset_id'],inplace = True)
        validation.set_index(['unit_id','dataset_id'],inplace = True)

        if test is not None:
            dataframe_HDBscan_test = pd.DataFrame(preprocess_HDBscan.transform(test['HDBScan']),
                                                  columns=setting_operational)
            dataframe_dataset_id_test = pd.DataFrame(preprocess_ID.transform(test.reset_index()['dataset_id']),
                                                      columns=dataset_id_columns)

            test = test.reset_index().join(dataframe_HDBscan_test)
            test = test.join(dataframe_dataset_id_test)
            
            test.set_index(['unit_id','dataset_id'],inplace = True)

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

    def op_factors(self,train, validation, test=None):
        #normalization for settings
        setting_train = train[settings].values
        setting_validation = validation[settings].values
        Scaler = StandardScaler().fit(setting_train)
        setting_train_normal = Scaler.transform(setting_train)
        setting_validation_normal = Scaler.transform(setting_validation)
        
        op_train_factors = []
        for i,data in enumerate(setting_train_normal):
            op_train_factors.append(((data[0]**2+data[1]**2+data[2]**2)/3)**0.5)
        train['op factor'] = op_train_factors
        
        op_validation_factors = []
        for i,data in enumerate(setting_validation_normal):
            op_validation_factors.append(((data[0]**2+data[1]**2+data[2]**2)/3)**0.5)
        validation['op factor'] = op_validation_factors
        
        if test is not None:
            setting_test = test[settings].values
            setting_test_normal = Scaler.transform(setting_test)
            op_test_factors = []
            for i,data in enumerate(setting_test_normal):
                op_test_factors.append(((data[0]**2+data[1]**2+data[2]**2)/3)**0.5)
            test['op factor'] = op_test_factors
            
            return train, validation, test
        else:
            return train, validation

    def prepare_datareader(self, batch_size, validation_split, number_steps_train, normalization):

        train_turbines, validation_turbines = train_test_split(self.train_turbines, test_size=validation_split)

        idx_train = self.train.index.to_series().unique()[train_turbines]
        idx_validation = self.train.index.to_series().unique()[validation_turbines]
        idx_test = self.test.index.to_series().unique()[self.test_turbines]
        
        train = self.train.loc[idx_train]
        validation = self.train.loc[idx_validation]
        test = self.test.loc[idx_test]
        
        if 'op factor' in self.feature_used:
            train,validation,test = self.op_factors(train, validation,test)
        else:
            pass
        
        ##one-hot coding
        if 'setting_op 1' in self.feature_used:
            train, validation, test = self.binarize(train, validation, test) #one-hot coding
        else:
            pass

        train, validation, test = self.normalize_by_type(train, validation, normalization, test)

        # ouly retain the feature we choose
        train = train[self.feature_used]
        validation = validation[self.feature_used]
        test = test[self.feature_used]

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