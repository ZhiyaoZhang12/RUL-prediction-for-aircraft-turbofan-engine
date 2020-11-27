# -*- coding: utf-8 -*-
"""
definination for error, score, and accuracy
"""

import os, datetime
import numpy as np
import pandas as pd

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

def error_function(df, y_predicted, y_true):
    return int(df[y_predicted] - df[y_true])

def score_function(df, label, alpha1=13, alpha2=10):      #defination for score
    if df[label] <= 0:
        return (np.exp(-(df[label] / alpha1)) - 1)

    elif df[label] > 0:
        return (np.exp((df[label] / alpha2)) - 1)

def accuracy_function(df, label, alpha1=13, alpha2=10):
    if df[label]<-alpha1 or df[label]>alpha2:
        return 0
    return 1