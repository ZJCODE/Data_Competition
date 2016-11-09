# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:56:53 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
import jieba
import string
import itertools
from collections import Counter
import random
import time

TestPath = './Data/user_tag_query_TEST.csv'

def SplitText(Sentence):
    return list(jieba.cut(Sentence, cut_all=False))

Pathstopwords = './Data/StopWords.txt'

def ReadStopWords(Pathstopwords):
    stopwords = []
    with open(Pathstopwords,'r') as f:
        Content = f.readlines()
        for line in Content:
            stopwords.append(line.strip().decode('gbk'))
    return stopwords

stopwords = ReadStopWords(Pathstopwords)

def ReadTestData(Path):
    ID = []    
    #QueryList = []
    QueryWordsList =[]
    with open(Path,'r') as f:
        Content = f.readlines()
        i=0
        for line in Content:
            t1 = time.time()
            linecontent = line.split('\t')
            ID.append(linecontent[0])   
            #Query = []
            QueryWordsWithPunctuaction = []
            for w in linecontent[1:]:
                try:
                    q =w.decode('gbk')
                    #Query.append(q)
                    QueryWordsWithPunctuaction += SplitText(q)
                    QueryWords = [w for w in QueryWordsWithPunctuaction if (w not in string.punctuation and w not in stopwords)]
                except:
                    pass
            #QueryList.append(Query)
            QueryWordsList.append(QueryWords)
            i=i+1
            t2 = time.time()
            print '=====add ' + str(i) +' lines Cost: '+ str(t2-t1)+'  seconds====='

    TestData = pd.DataFrame({'ID':ID,'QueryWordsList':QueryWordsList})
    return TestData
    

def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()
  

if __name__ == '__main__':
    Data = ReadTestData(TestPath)
    Save_Obj(Data,'DataTest')

