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

#DictNumTag = Import_Obj('DictNumTag')
#DictNumWordList = Import_Obj('DictNumWordList')
#TrainData = Import_Obj('DataTrain')
TestData = Import_Obj('DataTest')

CategoryCorpusJudge = [1 if len(a)>0 else 0 for a in DictNumWordList.values()]

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

#Save_Obj(AllWord,'AllWord')


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
        
  
def GetCategoryVectorElement():
    CategoryVectorElement = []
    for i in range(72):       
        t1= time.time()
        if CategoryCorpusJudge[i] > 0:            
            CategoryVectorElement.append(GetTFIDFVectorElement(DictNumWordList[i]))
        else:
            CategoryVectorElement.append(np.nan)
        t2= time.time()
        print i
        print t2-t1
    return CategoryVectorElement
      


CategoryVectorElement = GetCategoryVectorElement()

def SaveCategoryVectorElement():
    for i in range(72):
        pd.DataFrame(CategoryVectorElement[i].items()).to_csv('./Data/Category/' + str(i) + '.csv',encoding='utf8',header=False,index = False)


def ComputeCosine(V1,V2):
    V1Norm = np.linalg.norm(V1.values())
    V2Norm = np.linalg.norm(V2.values())
    Index = list((set(V1.keys()) & set(V2.keys())))
    V1dotV2 = sum([V1[i]*V2[i] for i in Index])
    Cosine = V1dotV2*1.0 / (V1Norm*V2Norm)
    return Cosine
    
Percent = pd.read_csv('Percent_tag.csv')
PercentTag = Percent.Percent

from math import sqrt
        
def sigmoid(x):
    return pow(x,1.0/3)

def Predict(corpus):

    Index = -1
    Vector_corpus  =GetTFIDFVectorElement(corpus)
    Cosine = -100
    for i in range(72):
        if CategoryCorpusJudge[i]>0:                
            Vector_Category = CategoryVectorElement[i]
            Cos = ComputeCosine(Vector_corpus,Vector_Category)*sigmoid(PercentTag[i])
            #print Cos
            if Cos > Cosine:
                Index = i
                Cosine = Cos
    return DictNumTag[Index]
        


PredictTag = []
i=0
for query in TestData.QueryWordsList:
    i = i+1
    t1= time.time()
    PredictTag.append(Predict(query))
    print i
    t2= time.time()
    print t2-t1
    
    
    
Age = [a[0] for a in PredictTag]
Gender = [a[2] for a in PredictTag]
Education = [a[1] for a in PredictTag]

PredictResult = pd.DataFrame({'ID':TestData.ID,'Age':Age,'Gender':Gender,'Education':Education})
Result = PredictResult[['ID','Age','Gender','Education']]
Result.to_csv('./Data/Predict.csv',sep = ' ', encoding='gbk',header=False,index = False)
