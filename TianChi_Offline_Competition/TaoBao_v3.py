# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:42:52 2016

@author: ZJun
"""

from Function_TaoBao import *

Data,Predict_Set = Load_Data()

DB,DP = Get_Data(Data,2014,12,9)
X,y = Get_X_y(DB,DP)


for i in range(10,19):
    DB,DP = Get_Data(Data,2014,12,i)
    X_1,y_1 = Get_X_y(DB,DP)
    X = np.vstack([X,X_1])
    y = np.hstack([y,y_1])
    
X_1 = X[y==1]
X_0 = X[y==0]

import random

X_0_sample = np.array(random.sample(X_0,len(X_1)*10))

XX = np.vstack([X_1,X_0_sample])
yy = np.array([1]*len(X_1)+[0]*len(X_0_sample))



temp,D_18 = DB,DP = Get_Data(Data,2014,12,18)
del temp
Attr_18,Index_18 = Get_Attr_New(D_18)
X_18 = np.nan_to_num(Attr_18)

Buy = Model(XX,yy,X_18,Index_18)
    
Predict_Set_item = set([str(i) for i in Predict_Set.item_id])

Buy_Predict = Buy[Buy.item_id.isin(Predict_Set_item)]

def save(df):
	df.to_csv('tianchi_mobile_recommendation_predict.csv', sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

save(Buy_Predict)

#----------------------

import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix