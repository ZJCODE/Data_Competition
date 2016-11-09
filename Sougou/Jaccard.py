# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:32:43 2016

@author: ZJun
"""

import pandas as pd
import time
import random



def Import_Obj(File):    
    import pickle
    File_Name = File+'.pkl'
    pkl_file = open(File_Name, 'rb')
    return  pickle.load(pkl_file)
    
    
def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()



TrainData = Import_Obj('DataTrain')
TestData = Import_Obj('DataTest')




def Jaccard(corpus1,corpus2):
    intersection = set(corpus1) & set(corpus2)
    union = set(corpus1) | set(corpus2)
    return len(intersection)*1.0 / len(union)
  
'''  
t1 = time.time()
j = Jaccard(TrainData.QueryWordsList[3],TestData.QueryWordsList[5])
t2 = time.time()
print t2-t1
'''



PredictTag = []
i = 0
for corpus1 in TestData.QueryWordsList:
    t1 = time.time()
    j = -1
    tagPre = '0'
    Random_Sample = random.sample(zip(TrainData.QueryWordsList,TrainData.ModifyTag),4000)
    for corpus2,tag in Random_Sample:
        jaccard = Jaccard(corpus1,corpus2)
        if jaccard > j:
            tagPre = tag
            j = jaccard
    PredictTag.append(tagPre)
    t2 = time.time()
    i = i+1
    print '====== Predict ' + str(i) +'th  ===  |  ===== Cost :' + str(t2-t1) + '  Seconds '
            
            
Age = [a[0] for a in PredictTag]
Gender = [a[2] for a in PredictTag]
Education = [a[1] for a in PredictTag]

PredictResult = pd.DataFrame({'ID':TestData.ID,'Age':Age,'Gender':Gender,'Education':Education})
Result = PredictResult[['ID','Age','Gender','Education']]
Result.to_csv('./Data/Predict.csv',sep = ' ', encoding='gbk',header=False,index = False)

    