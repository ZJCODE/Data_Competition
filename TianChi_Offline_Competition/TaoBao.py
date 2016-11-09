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

Predict_Set = pd.read_csv('./Data/tianchi_fresh_comp_train_item.csv')



# Get Before Predict Day's Data [t days ago]
Date = datetime(2014,12,18)

t = 1

Date_Before = Date - timedelta(t)

Date_Before_Data = Data[(Data.DateTime >= Date_Before) & (Data.DateTime < (Date_Before + timedelta(1)))]

Date_Predict_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]

Data = Date_Predict_Data

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



'''
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
'''
    

Date_Predict_Data_Buy = Date_Predict_Data[Date_Predict_Data.behavior_type == 4]
U_I = set([str(u)+'-'+str(i) for u,i in zip(Date_Predict_Data_Buy.user_id,Date_Predict_Data_Buy.item_id)])

y = np.array([1 if i in U_I else 0 for i in Index])
X = np.nan_to_num(Attr)


X_y_1 = X[y==1]
for i in range(20):
    X = np.vstack([X,X_y_1])
    
y_1 = y[y==1]
for i in range(20):    
    y = np.hstack([y,y_1])


#------------------------------------------------------
'''
Select_Date_Predict_Data = Data_Select(Date_Predict_Data)

Select = [1 if i in Select_Date_Predict_Data else 0 for i in Date_Predict_Data.user_item]
Date_Predict_Data['Select'] = Select
Date_Predict_Data = Date_Predict_Data[Date_Predict_Data.Select == 1]

Date_Predict_Data = Date_Predict_Data[[u'user_id', u'item_id', u'behavior_type', u'user_geohash',
       u'item_category', u'DateTime']]

'''

Attr_18,Index_18 = Get_Attr_New(Date_Predict_Data)
X_18 = np.nan_to_num(Attr_18)



'''
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X) # X 数据集   

X_reduced = pca.transform(X)
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y)
'''




# Model selection 
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
clf = SGDClassifier(loss="hinge", penalty="l2")
#loss : str, 'hinge', 'log', 'modified_huber', 'squared_hinge',
#'perceptron', or a regression loss: 'squared_loss', 'huber', 
#'epsilon_insensitive', or 'squared_epsilon_insensitive'
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
#-----------------------------------------------------------------
'''

# Model Evaluation
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

Predict_Set_item = set([str(i) for i in Predict_Set.item_id])

Buy_Predict = Buy_Predict[Buy_Predict.item_id.isin(Predict_Set_item)]


def save(df, path):
	df.to_csv('tianchi_mobile_recommendation_predict.csv', sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

