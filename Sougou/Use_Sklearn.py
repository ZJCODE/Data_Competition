# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:50:22 2016

@author: ZJun
"""

from sklearn.cross_validation import train_test_split
from sklearn import feature_extraction
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import metrics  
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
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

def load_data():
    TrainData = Import_Obj('DataTrain')
    TestData = Import_Obj('DataTest')
    Test = [' '.join(x) for x in TestData.QueryWordsList]
    Train = [' '.join(x) for x in TrainData.QueryWordsList]
    Train_y = TrainData.ModifyTag
    return Train,Train_y,Test



def get_word_feature(Train,Test):
    hv = HashingVectorizer(n_features=80000, non_negative=True)
    vectorizer = make_pipeline(hv, TfidfTransformer())
    train_feature = vectorizer.fit_transform(Train).toarray()
    test_feature = vectorizer.transform(Test) 
    return train_feature,test_feature


AllTag = list(itertools.product(*[('1','2','3','4','5','6'),('1','2','3','4','5','6'),('1','2')]))    
Tag_Dict = dict(zip(range(72),AllTag))
Tag_Dict_Reverse = dict(zip(AllTag,range(72)))


train_y = TrainData.Age.values

Train , Train_y , Test = load_data()

train_y = np.array([Tag_Dict_Reverse[y] for y in Train_y])


train_feature , test_feature = get_word_feature(Train,Test)


def pcaX(X):
    pca = PCA(n_components=1000)
    pca.fit(X) # X 数据集   
    X_reduced = pca.transform(X)
    return X_reduced
    


#del Train ,Test

X_train,X_test,y_train,y_test = train_test_split(train_feature,train_y)



from collections import Counter   

def Accuracy(pred,y_test):
    TrueFalse = []
    for i,j in zip(pred,y_test):        
        judge = np.array(list(Tag_Dict[i])) == np.array(list(Tag_Dict[j]))
        TrueFalse += list(judge)
    C = Counter(TrueFalse)
    accuracy = C.values()[1]*1.0/sum(C.values())
    return accuracy

def nbClassifier(X_train,X_test,y_train,y_test):

    clf = MultinomialNB(alpha = 0.01)   
    clf.fit(X_train,np.array(y_train))
    print '========Model Fitted=========='
    pred = clf.predict(X_test)
    print '========Predict Finished======'
    return Accuracy(y_test,pred)


def svmClassifier(X_train,X_test,y_train,y_test):
    
    svclf = SVC()#default with 'rbf'  
    svclf.fit(X_train,np.array(y_train))  
    print '========Model Fitted=========='
    pred = svclf.predict(X_test)
    print '========Predict Finished======'
    return Accuracy(y_test,pred)


def logistClassifier(X_train,X_test,y_train,y_test):
    
    Logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1000, tol=0.0008)
    Logistic.fit(X_train,np.array(y_train))  
    print '========Model Fitted=========='
    pred = Logistic.predict(X_test)
    print '========Predict Finished======'
    return Accuracy(y_test,pred)

def sgdClassifier(X_train,X_test,y_train,y_test):
    sgdlf = SGDClassifier(loss="hinge", penalty="l2")
    #loss : str, 'hinge', 'log', 'modified_huber', 'squared_hinge',
    #'perceptron', or a regression loss: 'squared_loss', 'huber', 
    #'epsilon_insensitive', or 'squared_epsilon_insensitive'
    t1 = time.time()
    sgdlf.fit(X_train,np.array(y_train))
    t2 = time.time()
    print '========Model Fitted==========Cost: '+str(t2-t1)+'seconds'
    pred = sgdlf.predict(X_test)
    t3 = time.time()
    print '========Predict Finished======Cost: '+str(t3-t2)+'seconds'
    return Accuracy(y_test,pred)

'''
kf = KFold(len(train_feature),n_folds=10)
score = 0
for train, test in kf:

    X_train, X_test = train_feature[train], train_feature[test]
    y_train, y_test = train_y[train], train_y[test]
    accuracy = nbClassifier(X_train,X_test,y_train,y_test)
    print accuracy
    score = score + accuracy
'''

def OutPut():
    TestID = Import_Obj('TestID')
    
    #Save_Obj(TestID,'TestID')  
    
    PredictTag = [Tag_Dict[i] for i in pre]
    
    Age = [a[0] for a in PredictTag]
    Gender = [a[2] for a in PredictTag]
    Education = [a[1] for a in PredictTag]
    
    PredictResult = pd.DataFrame({'ID':TestData.ID,'Age':Age,'Gender':Gender,'Education':Education})
    Result = PredictResult[['ID','Age','Gender','Education']]
    Result.to_csv('./Data/Predict.csv',sep = ' ', encoding='gbk',header=False,index = False)

    
    
