# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 22:25:45 2021

@author: Administrator
"""

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans
from sklearn import datasets 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv('./data/FD004/train_op.csv',header = 0,index_col = 'dataset_id')
print(data.columns)


data_nor = data
Scale = StandardScaler().fit(data)
data_nor = Scale.transform(data)



data_nor.to_csv('train_density.csv',header = True, index_col = True)
print('done!')