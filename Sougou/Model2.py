# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:19:20 2016

@author: ZJun
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 23:35:15 2016

@author: ZJun
"""

#import gc  
import pandas as pd
import numpy as np
import jieba
import string
import itertools
from collections import Counter
import random
import time


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


def GetWordInCorpusCount():
    Corpus1 = TrainData.QueryWordsList.values
    del TrainData
    gc.collect()  
    Corpus2  = TestData.QueryWordsList.values
    Corpus = np.hstack([Corpus1,Corpus2])
    del Corpus1,Corpus2
    gc.collect()  
        
    CombineAllSet = []
    for c in Corpus:
        CombineAllSet += list(set(c))
    WordInCorpusCount = Counter(CombineAllSet)
    
    return WordInCorpusCount


#AllWord = list(set(WordInCorpusCount.keys()))




def GetIDF(WordInCorpusCount,Corpus):
    Sum = len(Corpus)
    IDF = np.log(Sum*1.0 / (np.array(WordInCorpusCount.values())+1))
    return dict(zip(WordInCorpusCount.keys(),IDF))

#IDF_Dict = GetIDF(WordInCorpusCount,Corpus)

#Save_Obj(IDF_Dict,'IDF_Dict')


IDF_Dict = Import_Obj('IDF_Dict')
AllWord = Import_Obj('AllWord')

WordDict = dict(zip(AllWord,range(len(AllWord))))


def GetTF(corpus):
    C = Counter(corpus)
    Sum = sum(C.values())
    TF = np.array(C.values()) / (Sum * 1.0)
    TF_Dict = dict(zip(C.keys(),TF))
    return TF_Dict
  


def GetTFIDFVectorElement(corpus):
    Index = []
    TFIDF = []
    TF = GetTF(corpus)
    for w in TF.keys():
        Index.append(WordDict[w])
        TFIDF.append(TF[w]*IDF_Dict[w])
    return dict(zip(Index,TFIDF))
        
   

def ComputeCosine(V1,V2):
    V1Norm = np.linalg.norm(V1.values())
    V2Norm = np.linalg.norm(V2.values())
    Index = list((set(V1.keys()) & set(V2.keys())))
    V1dotV2 = sum([V1[i]*V2[i] for i in Index])
    Cosine = V1dotV2*1.0 / (V1Norm*V2Norm)
    return Cosine
    


        
Train_X =[]
i=0
for query in TrainData.QueryWordsList:
    Train_X.append(GetTFIDFVectorElement(query))
    i=i+1
    print i

Test_X = []
i=0
for query in TestData.QueryWordsList:
    Test_X.append(GetTFIDFVectorElement(query))
    i=i+1
    print i
    
Train_y = TrainData.ModifyTag.values




Train = zip(Train_X,Train_y)



import time
t1 = time.time()

Tag1 = []
#Tag2 = []
i=0
for test in Test_X[:2]:
    tag = '0'
    S = 100
    T=[]
    for train,y in random.sample(Train,1000):
        Sim = ComputeCosine(test,train)
        #T.append((y,Sim))
        if S > Sim:
            tag = y
            S = Sim
    Tag1.append(tag)
   # A = pd.DataFrame(T,columns=['tag','sim']).pivot_table(values = 'sim' , index = 'tag' ,aggfunc = 'sum')
    #Tag2.append(A.sort_values().index[-1])
    i=i+1
    print i

t2 = time.time()
print t2-t1

    
    
'''    
Age = [a[0] for a in PredictTag]
Gender = [a[2] for a in PredictTag]
Education = [a[1] for a in PredictTag]

PredictResult = pd.DataFrame({'ID':TestData.ID,'Age':Age,'Gender':Gender,'Education':Education})
Result = PredictResult[['ID','Age','Gender','Education']]
Result.to_csv('./Data/Predict.csv',sep = ' ', encoding='gbk',header=False,index = False)
'''