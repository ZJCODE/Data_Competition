# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:38:37 2016

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



'''
Data.columns
#[u'user_id', u'item_id', u'behavior_type', u'user_geohash',u'item_category', u'DateTime']

Num_Geo = len(set(Data.user_geohash))  # 1018982
Num_User = len(set(Data.user_id)) # 20000
Num_Category = len(set(Data.item_category)) # 9557
Num_item = len(set(Data.item_id)) # 4758484

Data_Start = min(Data.DateTime) # Timestamp('2014-11-18 00:00:00')
Date_End = max(Data.DateTime) # Timestamp('2014-12-18 23:00:00')

P = pd.read_csv('./Data/tianchi_fresh_comp_train_item.csv')
P.columns
# [u'item_id', u'item_geohash', u'item_category']

Num_Same_Geo = len(set(Data.user_geohash) & set(P.item_geohash)) # 25703





Dict_Item_Id_Geo = dict(zip(P.item_id,P.item_geohash))

Test_Data_item_geo = []

for item_id in Test_Data.item_id:    
    try:
        Test_Data_item_geo.append(Dict_Item_Id_Geo[item_id])
    except:
        Test_Data_item_geo.append(np.nan)
        
Test_Data['item_geohash'] = Test_Data_item_geo
'''




# Get Before Predict Day's Data [t days ago]
Date = datetime(2014,12,18)

t = 1

Date_Before = Date - timedelta(t)

Date_Before_Data = Data[(Data.DateTime >= Date_Before) & (Data.DateTime < (Date_Before + timedelta(1)))]

Date_Predict_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]

del Data


#------------------------------------------------------------------------------------

'''

def Get_User_Item_Data(Data,user_id,item_id = None ,behavior = None):
       
    if behavior is None:
        data = Data[Data.user_id == user_id]
    else:
        data = Data[(Data.user_id == user_id) & (Data.behavior_type == behavior)]
    sort_data = data.sort_values(by = 'DateTime')
    
    if item_id is None:
        return sort_data
    else:    
        UIdata = sort_data[sort_data.item_id == item_id]
        return UIdata
        
        

def Get_Similar_Data(Data,user_id,item_id,behavior = None):
    Dict_item_id_category = dict(set(zip(Data.item_id,Data.item_category)))
    if behavior is None:
        data = Data[Data.user_id == user_id]
    else:
        data = Data[(Data.user_id == user_id) & (Data.behavior_type == behavior)]
    Category = Dict_item_id_category[item_id]
    data = data[data.item_category == Category]
    return data
    


def Get_Attr(Date_Before_Data):
    User_Item_Id = [str(u)+'-'+str(i) for u,i in zip(Date_Before_Data.user_id,Date_Before_Data.item_id)]
    User_Item_Id_Set = list(set(User_Item_Id))[:20]
    Date_Before_Data['User_Item'] = User_Item_Id  # U-M 
    Attr = pd.DataFrame({'User_ID':User_Item_Id_Set})
    S_t_List = []
    A_t_List = []
    B_t_List = []
    SS_t_List = []
    SC_t_List = []
    SA_t_List = []
    SB_t_List = []
    AS_t_List = []
    AC_t_List = []
    AA_t_List = []
    AB_t_List = []
    
    for user_item in User_Item_Id_Set:
        u,i = [int(a) for a in user_item.split('-')]
        S_t = len(Get_User_Item_Data(Date_Before_Data,u,i,1))
        A_t = len(Get_User_Item_Data(Date_Before_Data,u,i,3))
        B_t = len(Get_User_Item_Data(Date_Before_Data,u,i,4))
        SS_t = len(Get_Similar_Data(Date_Before_Data,u,i,1))
        SC_t = len(Get_Similar_Data(Date_Before_Data,u,i,2))
        SA_t = len(Get_Similar_Data(Date_Before_Data,u,i,3))
        SB_t = len(Get_Similar_Data(Date_Before_Data,u,i,4))
        AS_t = len(Get_User_Item_Data(Date_Before_Data,u,behavior=1))
        AC_t = len(Get_User_Item_Data(Date_Before_Data,u,behavior=2))
        AA_t =len(Get_User_Item_Data(Date_Before_Data,u,behavior=3))
        AB_t = len(Get_User_Item_Data(Date_Before_Data,u,behavior=4))
        S_t_List.append(S_t)
        A_t_List.append(A_t)
        B_t_List.append(B_t)
        SS_t_List.append(SS_t)
        SC_t_List.append(SC_t)
        SA_t_List.append(SA_t)
        SB_t_List.append(SB_t)
        AS_t_List.append(AS_t)
        AC_t_List.append(AC_t)
        AA_t_List.append(AA_t)
        AB_t_List.append(AB_t)
        
    Attr['S_t'] = S_t_List
    Attr['A_t'] = A_t_List
    Attr['B_t'] = B_t_List
    Attr['SS_t'] = SS_t_List
    Attr['SC_t'] = SC_t_List
    Attr['SA_t'] = SA_t_List
    Attr['SB_t'] = SB_t_List
    Attr['AS_t'] = AS_t_List
    Attr['AC_t'] = AC_t_List
    Attr['AA_t'] = AA_t_List
    Attr['AB_t'] = AB_t_List
    
    return Attr
    

t1=time.time()
Attr = Get_Attr(Date_Before_Data)
t2 = time.time()
print t2-t1
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




import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

    

Date_Predict_Data_Buy = Date_Predict_Data[Date_Predict_Data.behavior_type == 4]
U_I = set([str(u)+'-'+str(i) for u,i in zip(Date_Predict_Data_Buy.user_id,Date_Predict_Data_Buy.item_id)])

y = np.array([1 if i in U_I else 0 for i in Index])
X = np.nan_to_num(Attr)

'''
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X) # X 数据集   

X_reduced = pca.transform(X)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
'''


Attr_18,Index_18 = Get_Attr_New(Date_Predict_Data)
X_18 = np.nan_to_num(Attr_18)


'''
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

#------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(Xtrain,ytrain)
ypred = rfc.predict(Xtest)


#-------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

#-------------------------------------------------------------------

from sklearn.svm import SVC
svm = SVC(kernel='linear')
#  It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'   #  算法的核
svm.fit(Xtrain,ytrain)  # X 数据集  y 类别变量
ypred = svm.predict(Xtest)

#------------------------------------------------------------------

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=2)

knn.fit(Xtrain,ytrain)       # X 数据集  y 类别变量
ypred = knn.predict(Xtest)

#------------------------------------------------------------------


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="squared_epsilon_insensitive", penalty="l2")
#loss : str, 'hinge', 'log', 'modified_huber', 'squared_hinge',
#'perceptron', or a regression loss: 'squared_loss', 'huber', 
#'epsilon_insensitive', or 'squared_epsilon_insensitive'
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
#-----------------------------------------------------------------
'''

'''
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(ytest, ypred)
Precision = metrics.precision_score(ytest, ypred)
Recall = metrics.recall_score(ytest, ypred)
F1 = metrics.f1_score(ytest, ypred)

confusion_matrix
'''


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X,y)
ypred = rfc.predict(X_18)

Result = pd.DataFrame({'User_Item':Index_18,'ypred':ypred})
Buy = Result[Result.ypred == 1].User_Item
user_id = [a.split('-')[0] for a  in Buy]
item_id = [a.split('-')[1] for a  in Buy]

Buy_Predict = pd.DataFrame({'user_id':user_id,'item_id':item_id})
Buy_Predict.to_csv('tianchi_mobile_recommendation_predict.csv',index=False, encoding='utf-8')

