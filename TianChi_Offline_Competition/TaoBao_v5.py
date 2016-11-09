
# Recall is too low

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
    

def Get_Data(Data,Y,M,D,t=1):
    Date = datetime(Y,M,D)
    Date_Before = Date - timedelta(t)
    Date_Before_Data = Data[(Data.DateTime >= Date_Before) & (Data.DateTime < (Date_Before + timedelta(1)))]
    Date_Predict_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]
    return Date_Before_Data,Date_Predict_Data


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
    Attr.index = Index
    Attr = Attr.fillna(0)
    Date_Predict_Data_Buy = Date_Predict_Data[Date_Predict_Data.behavior_type == 4]
    U_I = set([str(u)+'-'+str(i) for u,i in zip(Date_Predict_Data_Buy.user_id,Date_Predict_Data_Buy.item_id)])
    y = np.array([1 if i in U_I else 0 for i in Index])
    return Attr,y



def Choice_days_before(Data,Y,M,D,t,thread):
    DB,DP = Get_Data(Data,Y,M,D,t)  
    X,y= Get_X_y(DB,DP)
    X = X[[0,0,1,2,2,3,4,6,8,10]] 

    Mean_Buy = np.mean(X[y==1].values,0)
    Mean_Not_But = np.mean(X[y==0].values,0)
    Mean_All = np.mean(X.values,0)
    
    Mean = pd.DataFrame({'Buy':Mean_Buy,'Not_Buy':Mean_Not_But,'All':Mean_All})
    Judge = Mean.Buy > Mean.All
    
    DP_X,Index = Get_Attr_New(DP)
    DP_X.index = Index
    DP_X = DP_X.fillna(0)
    DP_X = DP_X[[0,0,1,2,2,3,4,6,8,10]] 
    
    For_Judge = DP_X.values > Mean.All.values
    T_count = [list(TF==Judge).count(True) for TF in For_Judge]
    Choice = DP_X[np.array(T_count) >= thread]
    return Choice


def Get_Date_Data(Data,Y,M,D):
    Date = datetime(Y,M,D)
    Date_Data = Data[(Data.DateTime >= Date) & (Data.DateTime < (Date + timedelta(1)))]
    return Date_Data




Data,Predict_Set = Load_Data()


def Choice_X(Data,Y,M,D):
    Day1_Choice = Choice_days_before(Data,Y,M,D,1,4)
    Day2_Choice = Choice_days_before(Data,Y,M,D,2,4)
    Day3_Choice = Choice_days_before(Data,Y,M,D,3,5)
    Choice = pd.concat([Day1_Choice,Day2_Choice,Day3_Choice])
    
    Choice['duplicate'] = Choice.index
    Choice = Choice.drop_duplicates('duplicate')
    del Choice['duplicate']
    '''
    Item = [int(a.split('-')[1]) for a in Choice.index]
    Choice['Item'] = Item
    Userful_Item = set(Predict_Set.item_id)
    Choice = Choice[Choice.Item.isin(Userful_Item)]
    del Choice['Item']
    '''
    return Choice

'''
Choice = Choice_X(Data,2014,12,17)    
    
Index = list(Choice.index)

D_18= Get_Date_Data(Data,2014,12,18)
D_Buy_18 = D_18[D_18.behavior_type==4]
u_i = set([str(i)+'-'+str(j) for i,j in zip(D_Buy_18.user_id,D_Buy_18.item_id)])

y = [1 if i in u_i else 0 for i in Index]
X = Choice.values


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
'''

def main(Data,Y,M,D):  
    
    def GetXy(Y,M,D):
        Choice = Choice_X(Data,Y,M,D)    
        Index = list(Choice.index)
        D= Get_Date_Data(Data,Y,M,D+1)
        D_Buy = D[D.behavior_type==4]
        u_i = set([str(i)+'-'+str(j) for i,j in zip(D_Buy.user_id,D_Buy.item_id)])
        y = [1 if i in u_i else 0 for i in Index]
        X = Choice.values
        return X,y
        
    X,y = GetXy(2014,12,10)
    
    for i in range(11,18):
        X1,y1=GetXy(2014,12,i)
        X = np.vstack([X,X1])
        y = np.hstack([y,y1])
    
    
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=0)
    rfc.fit(X,y)

    Choice_18 = Choice_X(Data,Y,M,D+1)
    X_18 = Choice_18.values
    ypred = rfc.predict(X_18)
    
    return Choice_18.index[ypred==1]
  
Buy = main(Data,2014,12,17)  
    
user_id = [a.split('-')[0] for a  in Buy]
item_id = [a.split('-')[1] for a  in Buy]

Buy_Predict = pd.DataFrame({'user_id':user_id,'item_id':item_id})

Predict_Set_item = set([str(i) for i in Predict_Set.item_id])

Buy_Predict = Buy_Predict[Buy_Predict.item_id.isin(Predict_Set_item)]


def save(df):
	df.to_csv('tianchi_mobile_recommendation_predict.csv', sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

    
    

