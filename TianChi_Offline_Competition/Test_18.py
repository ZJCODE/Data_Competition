# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:31:40 2016

@author: ZJun
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:42:52 2016

@author: ZJun
"""

from Function_TaoBao import *

Data,Predict_Set = Load_Data()



#-----------------------------------
DB,DP = Get_Data(Data,2014,12,8,1)
X,y, = Get_X_y(DB,DP)


for i in range(9,18):
    DB,DP = Get_Data(Data,2014,12,i,1)
    X_1,y_1= Get_X_y(DB,DP)
    X = pd.vstack([X,X_1])
    y = np.hstack([y,y_1])


    
#----------------------------------------------------


   
X_1 = X[y==1]
X_0 = X[y==0]

import random

X_0_sample = np.array(random.sample(X_0,len(X_1)*5))

XX = np.vstack([X_1,X_0_sample])
yy = np.array([1]*len(X_1)+[0]*len(X_0_sample))






D_17,D_18 = DB,DP = Get_Data(Data,2014,12,18)
Attr_17,Index_17 = Get_Attr_New(D_17)
X_17 = np.nan_to_num(Attr_17)


Buy = Model(XX,yy,X_17,Index_17)



#-----------------------------------------------------------

    
D = pd.merge(D_18,Predict_Set,on='item_id')

user_item = [str(i)+'-'+str(j) for i,j in zip(D.item_id,D.user_id)]
D['user_item'] = user_item
y = [1 if i==4 else 0 for i in D.behavior_type]
D['y'] = y
Buy_pred = [str(i)+'-'+str(j) for i,j in zip(Buy.item_id,Buy.user_id)]
ypred = [1 if i in set(Buy_pred) else 0 for i in D.user_item]


from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y, ypred)
Precision = metrics.precision_score(y, ypred)
Recall = metrics.recall_score(y, ypred)
F1 = metrics.f1_score(y, ypred)
