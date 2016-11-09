# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 22:31:31 2016

@author: ZJun
"""

#import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta


#conn= pymysql.connect(user='root', passwd='ZJMysql310')

Data = pd.read_csv('./Data/tianchi_fresh_comp_train_user.csv')
Data['DateTime']=pd.to_datetime(Data.time)
del Data['time']
#DateList = [t.date() for t in Data.DateTime]
#Data['Date'] = DateList


# Get Before Predict Day's Data [t days ago]
Date = datetime(2014,12,18)

t = 1

Date_Before = Date - timedelta(t)

Date_Before_Data = Data[(Data.DateTime >= Date_Before) & (Data.DateTime < (Date_Before + timedelta(1)))]

Date_Predict_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]

del Data


'''
def dell_df(df):
	df['num'] = [1]*len(df)
	user_item_id = [str(u)+'-'+str(i) for u,i in zip(df.user_id, df.item_id)]
	df['user_item'] =  user_item_id
	return df

def pivot_df(df):
	attr = df.pivot_table(values='num', index='user_item',columns='behavior_type', aggfunc=np.sum, fill_value=0)
	return attr
 
 
def chose_df(df):
	df = df[(df[1] >= 4) & (df[4] == 0) & (df[3] > 0)]
	df['user_item'] = df.index
	return df
 
 
def Data_Select(df):    
    DF_select = dell_df(df)
    DF_select = pivot_df(DF_select)
    DF_select = chose_df(DF_select)
    return set(DF_select.index)
    
Select_Date_Before_Data = Data_Select(Date_Before_Data)

Select = [1 if i in Select_Date_Before_Data else 0 for i in Date_Before_Data.user_item]
Date_Before_Data['Select'] = Select
Date_Before_Data = Date_Before_Data[Date_Before_Data.Select == 1]

Date_Before_Data = Date_Before_Data[[u'user_id', u'item_id', u'behavior_type', u'user_geohash',
       u'item_category', u'DateTime']]
'''

def Get_Attr_New(Date_Before_Data):
    Date_Before_Data['Num'] = [1]*len(Date_Before_Data)
    User_Item_Id = [str(u)+'-'+str(i) for u,i in zip(Date_Before_Data.user_id,Date_Before_Data.item_id)]
    Date_Before_Data['User_Item'] = User_Item_Id  # U-M 
    del User_Item_Id
    User_Category = [str(u)+'-'+str(i) for u,i in zip(Date_Before_Data.user_id,Date_Before_Data.item_category)]
    Date_Before_Data['User_Category'] = User_Category  # U-C
    del User_Category


    Attr_1 = Date_Before_Data.pivot_table('Num','User_Item','behavior_type',aggfunc = sum)
    Index = Attr_1.index
    
    Attr_1_1 = list(Attr_1.ix[:,1].values)
    Attr_1_2 = list(Attr_1.ix[:,2].values)
    Attr_1_3 = list(Attr_1.ix[:,3].values)
    Attr_1_4 = list(Attr_1.ix[:,4].values)
    
    #All
    Attr_2_Ready = Date_Before_Data.pivot_table('Num','user_id','behavior_type',aggfunc = sum)
    
    Attr_2_Dict_1 = dict(zip(Attr_2_Ready.index,Attr_2_Ready.ix[:,1]))
    Attr_2_Dict_2 = dict(zip(Attr_2_Ready.index,Attr_2_Ready.ix[:,2]))
    Attr_2_Dict_3 = dict(zip(Attr_2_Ready.index,Attr_2_Ready.ix[:,3]))
    Attr_2_Dict_4 = dict(zip(Attr_2_Ready.index,Attr_2_Ready.ix[:,4]))
    
    Index_user = [int(i.split('-')[0]) for i in Index]
    Attr_2_1 = [Attr_2_Dict_1[a] for a in Index_user]
    Attr_2_2 = [Attr_2_Dict_2[a] for a in Index_user]
    Attr_2_3 = [Attr_2_Dict_3[a] for a in Index_user]
    Attr_2_4 = [Attr_2_Dict_4[a] for a in Index_user]
    
    
    #Similar
    Attr_3_Ready = Date_Before_Data.pivot_table('Num','User_Category','behavior_type',aggfunc = sum)
    
    Attr_3_Dict_1 = dict(zip(Attr_3_Ready.index,Attr_3_Ready.ix[:,1]))
    Attr_3_Dict_2 = dict(zip(Attr_3_Ready.index,Attr_3_Ready.ix[:,2]))
    Attr_3_Dict_3 = dict(zip(Attr_3_Ready.index,Attr_3_Ready.ix[:,3]))
    Attr_3_Dict_4 = dict(zip(Attr_3_Ready.index,Attr_3_Ready.ix[:,4]))

    Item_Id_Category_Dict = dict(zip(Date_Before_Data.item_id,Date_Before_Data.item_category))
    Index_User_Category = [i.split('-')[0] +'-'+ str(Item_Id_Category_Dict[int(i.split('-')[1])]) for i in Index]
    
    Attr_3_1 = [Attr_3_Dict_1[a] for a in Index_User_Category]
    Attr_3_2 = [Attr_3_Dict_2[a] for a in Index_User_Category]
    Attr_3_3 = [Attr_3_Dict_3[a] for a in Index_User_Category]
    Attr_3_4 = [Attr_3_Dict_4[a] for a in Index_User_Category]
    
    Attr = pd.DataFrame(np.matrix([Attr_1_1,Attr_1_2,Attr_1_3,Attr_1_4,Attr_2_1,Attr_2_2,Attr_2_3,Attr_2_4,Attr_3_1,Attr_3_2,Attr_3_3,Attr_3_4]).T)
    
    return Attr,Index
    

    

    
import time

t1=time.time()
Attr,Index = Get_Attr_New(Date_Before_Data)
t2 = time.time()
print t2-t1



    

Date_Predict_Data_Buy = Date_Predict_Data[Date_Predict_Data.behavior_type == 4]
U_I = set([str(u)+'-'+str(i) for u,i in zip(Date_Predict_Data_Buy.user_id,Date_Predict_Data_Buy.item_id)])

y = np.array([1 if i in U_I else 0 for i in Index])
X = np.nan_to_num(Attr)



Attr_18,Index_18 = Get_Attr_New(Date_Predict_Data)
X_18 = np.nan_to_num(Attr_18)


mean = np.mean(X[y==1],0)


from collections import Counter

TF = X>mean

TF_Count = [Counter(TF[i,:]).values() for i in range(len(TF))]

# TF_Count = np.nan_to_num(TF_Count)

DF = pd.DataFrame(np.nan_to_num(pd.DataFrame(TF_Count)),columns=['False','True'])
DF['y'] = y


import matplotlib.pyplot as plt

#plt.scatter(DF.False,DF.True,c = DF.y,alpha = 0.2)


y_pred = [1 if i > 3.2 else 0 for i in DF.True]
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y, y_pred)
confusion_matrix