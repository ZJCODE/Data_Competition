# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 18:27:12 2016

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

TrainPath = './Data/user_tag_query_TRAIN.csv'

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

def ReadTrainData(Path):
    ID = []    
    Age = []
    Gender = []
    Education = []    
    #QueryList = []
    QueryWordsList =[]
    with open(Path,'r') as f:
        Content = f.readlines()
        i=0
        for line in Content:
            t1  =time.time()
            linecontent = line.split('\t')
            ID.append(linecontent[0])
            Age.append(linecontent[1])
            Gender.append(linecontent[2])
            Education.append(linecontent[3])       
            #Query = []
            QueryWordsWithPunctuaction = []
            for w in linecontent[4:]:
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
            print '=====add ' + str(i) +' lines of Train Cost: '+ str(t2-t1)+'  seconds====='
    #TrainData = pd.DataFrame({'ID':ID,'Age':Age,'Gender':Gender,'Education':Education,'QueryList':QueryList,'QueryWordsList':QueryWordsList})
    TrainData = pd.DataFrame({'ID':ID,'Age':Age,'Gender':Gender,'Education':Education,'QueryWordsList':QueryWordsList})
    return TrainData

TestPath = './Data/user_tag_query_TEST.csv'

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
            print '=====add ' + str(i) +' lines of Test Cost: '+ str(t2-t1)+'  seconds====='

    TestData = pd.DataFrame({'ID':ID,'QueryWordsList':QueryWordsList})
    return TestData
    

'''

Counter(Data.Age)
Out[142]: 
Counter({'0': 355,
         '1': 7900,
         '2': 5330,
         '3': 3603,
         '4': 2141,
         '5': 589,
         '6': 82})


Counter(Data.Education)
Out[143]: 
Counter({'0': 1878,
         '1': 65,
         '2': 119,
         '3': 3722,
         '4': 5579,
         '5': 7487,
         '6': 1150})

Counter(Data.Gender)
Out[144]: Counter({'0': 424, '1': 11365, '2': 8211})

Education      0     1     2       3       4       5      6
Age                                                        
0           29.0   NaN   NaN    11.0    14.0     2.0  299.0
1          570.0   1.0   3.0    44.0   479.0  6042.0  761.0
2          378.0   1.0   7.0   184.0  3315.0  1413.0   32.0
3          360.0  27.0  45.0  2096.0  1038.0    17.0   20.0
4          311.0  29.0  54.0  1209.0   498.0     8.0   32.0
5          175.0   7.0   9.0   172.0   218.0     5.0    3.0
6           55.0   NaN   1.0     6.0    17.0     NaN    3.0

Data.pivot_table('ID','Gender','Education',aggfunc='count')
Out[157]: 
Education       0     1     2       3       4       5      6
Gender                                                      
0            95.0   NaN   1.0    24.0    19.0    42.0  243.0
1          1058.0  39.0  61.0  1923.0  3167.0  4652.0  465.0
2           725.0  26.0  57.0  1775.0  2393.0  2793.0  442.0


Counter([(i,j,k) for i,j,k in zip(Data.Age,Data.Education,Data.Gender)])

# 预测共72种输出，统计算出72个类别向量，新的数据与这些类别向量计算角度

其中类似于(0，a，b)这样的按比例随机放入(1,a,b),(2,a,b)...(6,a,c) 

其中类似于(0,0,1)这样的数据直接丢弃

Age                   0       1       2       3      4      5     6
Gender Education                                                   
0      0            NaN    38.0    16.0    16.0    8.0    3.0  14.0
       2            NaN     NaN     NaN     1.0    NaN    NaN   NaN
       3            NaN     1.0     2.0    10.0   11.0    NaN   NaN
       4            NaN     2.0     9.0     6.0    2.0    NaN   NaN
       5            NaN    30.0    10.0     1.0    1.0    NaN   NaN
       6          196.0    33.0     8.0     2.0    4.0    NaN   NaN
1      0           20.0   338.0   188.0   194.0  183.0  111.0  24.0
       1            NaN     NaN     1.0    21.0   14.0    3.0   NaN
       2            NaN     NaN     4.0    18.0   32.0    6.0   1.0
       3            7.0    18.0    93.0  1105.0  602.0   95.0   3.0
       4            9.0   303.0  1811.0   658.0  266.0  114.0   6.0
       5            2.0  3847.0   789.0    10.0    2.0    2.0   NaN
       6           59.0   371.0    11.0     9.0   12.0    NaN   3.0
2      0            9.0   194.0   174.0   150.0  120.0   61.0  17.0
       1            NaN     1.0     NaN     6.0   15.0    4.0   NaN
       2            NaN     3.0     3.0    26.0   22.0    3.0   NaN
       3            4.0    25.0    89.0   981.0  596.0   77.0   3.0
       4            5.0   174.0  1495.0   374.0  230.0  104.0  11.0
       5            NaN  2165.0   614.0     6.0    5.0    3.0   NaN
       6           44.0   357.0    13.0     9.0   16.0    3.0   NaN

'''        

def AddTag(Data):
    Tag = [(i,j,k) for i,j,k in zip(Data.Age,Data.Education,Data.Gender)]
    Data['Tag'] = Tag
    return Data


def GenerateRandomNum(Data,Attr):
    L=[i for i in Data[Attr] if i != '0']
    return random.choice(L)


def ModifyTag(Data):
    ValidTag = []
    ModifiedTag = []
    AttrDict = {0:'Age',1:'Education',2:'Gender'}
    for tag in Data.Tag:
        CountZero = tag.count('0')
        if CountZero > 1:
            ValidTag.append(0)
            ModifiedTag.append(tag)
        else:
            ValidTag.append(1)
            if CountZero == 1:                
                Index = tag.index('0')
                Num = GenerateRandomNum(Data,AttrDict[Index])
                tagtolist = list(tag)
                tagtolist[Index] = Num
                ModifiedTag.append(tuple(tagtolist))
            else:
                ModifiedTag.append(tag)
    Data['ModifyTag'] = ModifiedTag
    Data['ValidTag'] = ValidTag
    DataValid = Data[Data.ValidTag == 1]
    del DataValid['ValidTag']
    DataValid.index = range(len(DataValid))
    return DataValid
    

                
def GetTagWordListDict(Data):    
    AllTag = list(itertools.product(*[('1','2','3','4','5','6'),('1','2','3','4','5','6'),('1','2')]))    
    AllWordList = [[]]*72    
    TagWordListDict = dict(zip(AllTag,AllWordList))

    for i in range(len(Data)):
        TagWordListDict[Data.ModifyTag[i]] = TagWordListDict[Data.ModifyTag[i]] + Data.QueryWordsList[i]
    
    DictNumTag = dict(zip(range(len(AllTag)),TagWordListDict.keys()))
    DictNumWordList = dict(zip(range(len(AllTag)),TagWordListDict.values()))
    
    return DictNumTag , DictNumWordList


def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()

if __name__ == '__main__':
    print 'Start Load Train Data'
    TrainData = ReadTrainData(TrainPath)
    print 'Train Data Loaded'
    TrainData = AddTag(TrainData)
    TrainData = ModifyTag(TrainData)
    print 'Tag Modified'
    #DictNumTag,DictNumWordList = GetTagWordListDict(Data)
    Save_Obj(TrainData,'DataTrain')
    print 'Start Load Test Data'
    TestData = ReadTestData(TestPath)
    Save_Obj(TestData,'DataTest')
    #Save_Obj(DictNumTag,'DictNumTag')
    #Save_Obj(DictNumWordList,'DictNumWordList')
    
