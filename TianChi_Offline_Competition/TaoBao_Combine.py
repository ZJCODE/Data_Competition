# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:53:12 2016

@author: ZJun
"""

import glob
import pandas as pd
import numpy as np
Results = glob.glob(r'./Result/*.csv')

Buy = pd.read_csv(Results[0],sep='\t')

for result in Results[1:]:
    Buy1 = pd.read_csv(result,sep='\t')
    Buy = pd.concat([Buy,Buy1])
    
Buy_user_item = [str(i)+'-'+str(j) for i,j in zip(Buy.user_id,Buy.item_id)]


from collections import Counter

Count_Buy = Counter(Buy_user_item)
'''
def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L
    
Sort_Count_Buy = Sort_Dict(Count_Buy)
'''

user_item = np.array(Count_Buy.keys())
count = np.array(Count_Buy.values())

Buy_Predict = user_item[count>=6]

user_id = [a.split('-')[0] for a  in Buy_Predict]
item_id = [a.split('-')[1] for a  in Buy_Predict]

Buy_Predict_Result = pd.DataFrame({'user_id':user_id,'item_id':item_id})

def save(df):
	df.to_csv('tianchi_mobile_recommendation_predict.csv', sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

save(Buy_Predict_Result)