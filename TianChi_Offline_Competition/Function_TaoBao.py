# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:45:42 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
#import seaborn
#seaborn.set()


def Load_Data():
    Data = pd.read_csv('./Data/tianchi_fresh_comp_train_user.csv')
    Data['DateTime']=pd.to_datetime(Data.time)
    del Data['time']
    #Day = [date.date() for date in Data.DateTime]
    #Hour = [date.hour for date in Data.DateTime]
    Predict_Set = pd.read_csv('./Data/tianchi_fresh_comp_train_item.csv')
    return Data,Predict_Set
    
def Load_Data_Merged():
    Data = pd.read_csv('./Data/tianchi_fresh_comp_train_user.csv')
    Data['DateTime']=pd.to_datetime(Data.time)
    del Data['time']
    #Day = [date.date() for date in Data.DateTime]
    #Hour = [date.hour for date in Data.DateTime]
    Predict_Set = pd.read_csv('./Data/tianchi_fresh_comp_train_item.csv')
    Data = pd.merge(Data,Predict_Set,on='item_id')
    return Data,Predict_Set
    
def Get_Data(Data,Y,M,D,t=1):
    Date = datetime(Y,M,D)
    Date_Before = Date - timedelta(t)
    Date_Before_Data = Data[(Data.DateTime >= Date_Before) & (Data.DateTime < (Date_Before + timedelta(1)))]
    Date_Predict_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]
    return Date_Before_Data,Date_Predict_Data
    
    
def Buy_Or_Not(Date_Before_Data,Date_Predict_Data):
    DB_user_item = [str(u)+'-'+str(i) for u,i in zip(Date_Before_Data.user_id,Date_Before_Data.item_id)]
    DP_user_item = [str(u)+'-'+str(i) for u,i in zip(Date_Predict_Data.user_id,Date_Predict_Data.item_id)]
    Date_Before_Data['user_item'] = DB_user_item
    Date_Predict_Data['user_item'] = DP_user_item
    Date_Predict_Data_Buy = Date_Predict_Data[Date_Predict_Data.behavior_type == 4]
    Buy_user_item = set(Date_Predict_Data_Buy.user_item)
    buy_or_not = [1 if u_i in Buy_user_item else 0 for u_i in Date_Before_Data.user_item]
    Date_Before_Data['buy_or_not'] = buy_or_not
    return Date_Before_Data
    
def Get_Hour(Data):
    Hour = [date.hour for date in Data.DateTime]
    Data['hour'] = Hour
    return Data
    

def Get_DB(Data,Y,M,D,t=1):
    Date_Before_Data,Date_Predict_Data = Get_Data(Data,Y,M,D,t=1)
    Date_Before_Data = Buy_Or_Not(Date_Before_Data,Date_Predict_Data)
    Date_Before_Data = Get_Hour(Date_Before_Data)
    return Date_Before_Data
    
'''
DB = Get_DB(Data,2014,12,9)

for i in range(10,19):
    DB = pd.concat([DB,Get_DB(Data,2014,12,i)])
'''
    
def Draw_Hour_Distribution(DB):   
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))   # 构建绘图框架
    fig.subplots_adjust(hspace=0.5, wspace=0.3)    
    plt.subplot(2,1,1)
    plt.hist(DB[DB.buy_or_not ==1].hour,color='r',alpha=0.3,bins=24)
    plt.title('Buy')
    plt.subplot(2,1,2)
    plt.hist(DB[DB.buy_or_not ==0].hour,color='b',alpha=0.3,bins=24)
    plt.title('Not_Buy')
    
    
def Draw_Hour_Distribution_Behavior(DB,behavior):   
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))   # 构建绘图框架
    fig.subplots_adjust(hspace=0.5, wspace=0.3)    
    plt.subplot(2,1,1)
    plt.hist(DB[(DB.behavior_type==behavior) & (DB.buy_or_not==1)].hour,color='r',alpha=0.3,bins=24)
    plt.title('Buy'+'  Behavior: '+str(behavior))
    plt.subplot(2,1,2)
    plt.hist(DB[(DB.behavior_type==behavior) & (DB.buy_or_not==0)].hour,color='b',alpha=0.3,bins=24)
    plt.title('Not_Buy'+'  Behavior: '+str(behavior))
    plt.savefig('Behavior['+str(behavior)+'] Hour Distribution')
 
def Draw_Buyer_Behavior(DB):    
    DB_Buy = DB[DB.buy_or_not==1]
    DB_Buy['Num'] = [1]*len(DB_Buy)
    hour_behavior = DB_Buy.pivot_table('Num','hour','behavior_type',aggfunc='sum')
    hour_behavior.plot(kind='bar',alpha=0.3)
    plt.title('Buyer\'s Behavior')
    
    
    

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
    

    
def Get_X_y(Date_Before_Data,Date_Predict_Data):    
    Attr,Index = Get_Attr_New(Date_Before_Data)
    X = np.nan_to_num(Attr)
    Date_Predict_Data_Buy = Date_Predict_Data[Date_Predict_Data.behavior_type == 4]
    U_I = set([str(u)+'-'+str(i) for u,i in zip(Date_Predict_Data_Buy.user_id,Date_Predict_Data_Buy.item_id)])
    y = np.array([1 if i in U_I else 0 for i in Index])
    return X,y



def Pca_X(X,n):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(X) # X 数据集   
    X_reduced = pca.transform(X)
    return X_reduced


def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()


def Test_Model(X,y):
    
    from sklearn.cross_validation import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(Xtrain,ytrain)
    ypred = rfc.predict(Xtest)
    
    from sklearn import metrics

    confusion_matrix = metrics.confusion_matrix(ytest, ypred)
    Precision = metrics.precision_score(ytest, ypred)
    Recall = metrics.recall_score(ytest, ypred)
    F1 = metrics.f1_score(ytest, ypred)
    print confusion_matrix
    return F1,Precision,Recall

def Model(X,y,Xpred,Index_pred):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X,y)
    ypred = rfc.predict(Xpred)
    Result = pd.DataFrame({'user_item':Index_pred,'ypred':ypred})
    Buy = Result[Result.ypred == 1].user_item
    user_id = [a.split('-')[0] for a  in Buy]
    item_id = [a.split('-')[1] for a  in Buy]
    Buy_Predict = pd.DataFrame({'user_id':user_id,'item_id':item_id})
    return Buy_Predict