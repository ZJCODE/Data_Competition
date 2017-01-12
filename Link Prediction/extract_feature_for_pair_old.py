#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:02:28 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from gensim import corpora, models, similarities
import networkx as nx
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn import svm


import nltk

stpwds = set(nltk.corpus.stopwords.words("english"))



def Import_Obj(File):    
    import pickle
    File_Name = File
    pkl_file = open(File_Name, 'rb')
    return  pickle.load(pkl_file)

    
    
train = pd.read_table('./Data/training_set.txt',sep=' ',names=['source','target','link'])
test = pd.read_table('./Data/testing_set.txt',sep=' ',names=['source','target'])
node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

G = nx.Graph()
train_link = train[train.link == 1][['source','target']]
G.add_edges_from(train_link.values)

author_split = []
for au in node_information.author:
    try:
        author_split.append(au.split(','))
    except:
        author_split.append(au)

node_information['author_split'] = author_split


abstract_feature  = pd.read_csv('./Data/abstract_feature.csv')
title_feature = pd.read_csv('./Data/title_feature.csv')

id_abstract_vector_dict = dict(zip(abstract_feature.id.values,abstract_feature.ix[:,1:].values))
id_title_vector_dict = dict(zip(title_feature.id.values,title_feature.ix[:,1:].values))

id_bc_dict = Import_Obj('./Data/id_bc_dict.pkl')
id_degree_dict = Import_Obj('./Data/id_degree_dict.pkl')
id_network_community_category_dict = Import_Obj('./Data/papers_network_community_category_dict.pkl')
id_cluster_dict  = Import_Obj('./Data/id_cluster_dict.pkl')
id_pagerank_dict = Import_Obj('./Data/id_pagerank_dict.pkl')


author_journal_feature = pd.read_csv('./Data/author_journal_feature_without_nan.csv')

id__author_paper_year_mean_dict = dict(zip(author_journal_feature.id.values,author_journal_feature.author_paper_year_mean.values))
id__author_paper_num_year_dict = dict(zip(author_journal_feature.id.values,author_journal_feature.author_paper_num_year.values))
id__journal_paper_num_year_dict = dict(zip(author_journal_feature.id.values,author_journal_feature.journal_paper_num_year.values))
id__journal_class_dict = dict(zip(author_journal_feature.id.values,author_journal_feature.journal_class.values))

id_title_dict = dict(zip(node_information.id.values,node_information.title.values))
id_year_dict = dict(zip(node_information.id.values,node_information.year.values))
id_author_split_dict = dict(zip(node_information.id.values,author_split))


def cosine(v1,v2):
    return np.dot(v1,v2)*1.0 / (np.linalg.norm(v1)*np.linalg.norm(v2))



def get_all_feature(relation):
    
    title_sim = []
    abstract_sim = []
    comm_word_title = []
    delta_year = []
    comm_author = []
    bc_max = []
    degree_max =[]
    paper_category = []

    author_paper_year_mean = []
    author_paper_num_year = []
    journal_paper_num_year  =[]
    journal_class = []

    short_path_feature = []
    cluster_max = []
    pagerank_max = []




    
    
    for r in relation:
        source,target = r
        
        try:            
            short_path_feature.append(nx.shortest_path_length(G,source,target))
        except:
            short_path_feature.append(0)

        try:
            cluster_max.append(max(id_cluster_dict[source],id_cluster_dict[target]))
        except:
            cluster_max.append(0)

        try:
            pagerank_max.append(max(id_pagerank_dict[source],id_pagerank_dict[target]))
        except:
            pagerank_max.append(0)


        try:
            t_s = cosine(id_title_vector_dict[source],id_title_vector_dict[target])
            if np.isnan(t_s):
                title_sim.append(0)
            else:                
                title_sim.append(t_s)
        except:
            title_sim.append(0)
            
        try:      
            a_s = cosine(id_abstract_vector_dict[source],id_abstract_vector_dict[target])
            if np.isnan(a_s):
                abstract_sim.append(0)
            else:                
                abstract_sim.append(a_s)
        except:
            abstract_sim.append(0)
            
        try:
            source_title = [w for w in id_title_dict[source].lower().split() if w not in stpwds]
            target_title = [w for w in id_title_dict[target].lower().split() if w not in stpwds]
            comm_word_title.append(len(set(source_title)&set(target_title)))
        except:
            comm_word_title.append(0)
            
        try:
            delta_year.append(abs(id_year_dict[source] - id_year_dict[target]))
        except:
            delta_year.append(0)
            
        try:
            comm_author.append(len(set(id_author_split_dict[source])&set(id_author_split_dict[target])))
        except:
            comm_author.append(0)
            
        try:
            bc_max.append(max(id_bc_dict[source],id_bc_dict[target]))
        except:
            bc_max.append(0)
            
        try:
            degree_max.append(max(id_degree_dict[source],id_degree_dict[target]))
        except:
            degree_max.append(0)
            
        try:
            if id_network_community_category_dict[source] == id_network_community_category_dict[target]:
                paper_category.append(1)
            else:
                paper_category.append(0)
        except:
            paper_category.append(0)

        try:
            author_paper_year_mean.append(max(id__author_paper_year_mean_dict[source],id__author_paper_year_mean_dict[target]))
        except:
            author_paper_year_mean.append(0)

        try:
            author_paper_num_year.append(max(id__author_paper_num_year_dict[source],id__author_paper_num_year_dict[target]))
        except:
            author_paper_num_year.append(0)

        try:
            journal_paper_num_year.append(max(id__journal_paper_num_year_dict[source],id__journal_paper_num_year_dict[target]))
        except:
            journal_paper_num_year.append(0)

        try:
            if id__journal_class_dict[source] == id__journal_class_dict[target]:
                journal_class.append(1)
            else:
                journal_class.append(0)
        except:
            journal_class.append(0)

                
    all_feature = np.array([title_sim,abstract_sim,comm_word_title,delta_year,comm_author,bc_max,degree_max,cluster_max,pagerank_max,short_path_feature,paper_category,author_paper_year_mean,author_paper_num_year,journal_paper_num_year,journal_class]).T
    

    return all_feature
    

relation_train = train[['source', 'target']].values
relation_test = test[['source', 'target']].values



feature_train = get_all_feature(relation_train)
feature_test = get_all_feature(relation_test)

def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()

Save_Obj(feature_train,'./Data/feature_train')
Save_Obj(feature_test,'./Data/feature_test')

print 'feature extracted'

training_features = preprocessing.scale(feature_train)
test_features = preprocessing.scale(feature_test)


label_train = train.link.values


from sklearn.cross_validation import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(training_features, label_train)

X_train, X_test, y_train, y_test = train_test_split(feature_train, label_train)



from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC 




def KnnClassifier(X_train,X_test,y_train,y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train,np.array(y_train))
    print '========Model Fitted========== '
    pred = knn.predict(X_test)
    print '========Predict Finished====== '
    print f1_score(pred,y_test)

#print 'knn: '
#KnnClassifier(X_train,X_test,y_train,y_test)



def logistClassifier(X_train,X_test,y_train,y_test):
    Logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1000, tol=0.0008)
    #Logistic = LogisticRegression(penalty='l2')
    Logistic.fit(X_train,np.array(y_train))  
    print '========Model Fitted========== '
    pred = Logistic.predict(X_test)
    print '========Predict Finished====== '
    print f1_score(pred,y_test)

print 'logist: '
#logistClassifier(X_train,X_test,y_train,y_test)


def rfClassifier(X_train,X_test,y_train,y_test):
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train,np.array(y_train))
    print '========Model Fitted=========='
    pred = rfc.predict(X_test)
    print '========Predict Finished======'
    print f1_score(pred,y_test)    


print 'random forest: '
#rfClassifier(X_train,X_test,y_train,y_test)

    
def gbClassifierPred(X_train, X_test, y_train,y_test):
    params = {'n_estimators': 300, 'max_depth': 5, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    print '========Model Fitted=========='
    pred = clf.predict(X_test)
    print '========Predict Finished======'
    print f1_score(pred,y_test)    

 
print 'gbdt: '
#gbClassifierPred(X_train, X_test, y_train,y_test)

def svmClassifierPred(X_train, X_test, y_train,y_test):
    svmclf = SVC(kernel='linear') #default with 'rbf'      
    svmclf.fit(X_train, y_train)
    print '========Model Fitted=========='
    pred = svmclf.predict(X_test)
    print '========Predict Finished======'
    print f1_score(pred,y_test)    

print 'svm: '
#svmClassifierPred(X_train, X_test, y_train,y_test)




def combine(X_train, X_test, y_train,y_test):
    Logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1000, tol=0.0008)
    Logistic.fit(X_train,np.array(y_train))  
    print '========Model 1 Fitted========== '
    pred1 = Logistic.predict(X_test)
    
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train,np.array(y_train))
    print '========Model 2 Fitted=========='
    pred2 = rfc.predict(X_test)
    
    params = {'n_estimators': 300, 'max_depth': 5, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    print '========Model 3 Fitted=========='
    pred3 = clf.predict(X_test)
    
    pred = (pred1+pred2+pred3)/2
    
    print '========Predict Finished======'
    print f1_score(pred,y_test)   
    
    
print 'combine: '
#combine(X_train, X_test, y_train,y_test)    
    

def Predict_combine(X_train, X_test, y_train):

    Logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1000, tol=0.0008)
    Logistic.fit(X_train,np.array(y_train))  
    print '========Model 1 Fitted========== '
    pred1 = Logistic.predict(X_test)
    
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train,np.array(y_train))
    print '========Model 2 Fitted=========='
    pred2 = rfc.predict(X_test)
    
    params = {'n_estimators': 300, 'max_depth': 5, 'subsample': 0.3,
              'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)
    print '========Model 3 Fitted=========='
    pred3 = clf.predict(X_test)
    
    pred = (pred1+pred2+pred3)/2

    return pred

'''
pred = Predict_combine(feature_train, feature_test, label_train)
submission = pd.DataFrame({'Id':range(len(pred)),'prediction':pred})
submission.to_csv('submission_combine.csv',header=True,index= False)
'''



'''
Logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1000, tol=0.0008)
Logistic.fit(training_features,np.array(label_train))  
pred = Logistic.predict(test_features)
'''

'''
classifier = svm.LinearSVC()
classifier.fit(training_features,np.array(label_train))  
pred = classifier.predict(test_features)
'''

'''
params = {'n_estimators': 300, 'max_depth': 5, 'subsample': 0.3,'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(training_features, label_train)
pred = clf.predict(test_features)
'''



rfc = RandomForestClassifier(n_estimators=200, random_state=0)
rfc.fit(feature_train,np.array(label_train))  
pred = rfc.predict(feature_test)


submission = pd.DataFrame({'Id':range(len(pred)),'prediction':pred})
submission.to_csv('submission_20170108.csv',header=True,index= False)

