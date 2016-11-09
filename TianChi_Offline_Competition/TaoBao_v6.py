# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 19:05:03 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt


def Load_Data():
    Data = pd.read_csv('./Data/tianchi_fresh_comp_train_user.csv')
    Data['DateTime']=pd.to_datetime(Data.time)
    del Data['time']
    #Day = [date.date() for date in Data.DateTime]
    #Hour = [date.hour for date in Data.DateTime]
    Predict_Set = pd.read_csv('./Data/tianchi_fresh_comp_train_item.csv')
    return Data,Predict_Set

def Get_Date_Data(Data,Y,M,D):
    Date = datetime(Y,M,D)
    Date_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]
    return Date_Data

def save(df):
	df.to_csv('tianchi_mobile_recommendation_predict.csv', sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

