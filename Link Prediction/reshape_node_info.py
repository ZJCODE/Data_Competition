import pandas as pd
import numpy as np

node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

author_split = []
for au in node_information.author:
    try:
        author_split.append(au.split(','))
    except:
        author_split.append(au)
node_information['author_split'] = author_split


node_information_ = node_information[['id','year','title','author','journal','abstract']].head(0)

i =0
info = node_information.ix[i,:]
info_ = list(info[['id','year','journal']].values)
author_length = len(info.author_split)
author_split = np.array(info.author_split)
author_length
info_s = [] 
for i in range(author_length):
    info_s.append(info_)
Data = np.hstack([info_s,author_split[:,np.newaxis]])


for i in range(1,len(node_information)):
    print 'process : ' +str(i) +' th'
    info = node_information.ix[i,:]
    info_ = list(info[['id','year','journal']].values)
    try:        
        author_length = len(info.author_split)
        author_split = np.array(info.author_split)
        author_length
        info_s = [] 
        for i in range(author_length):
            info_s.append(info_)
        data = np.hstack([info_s,author_split[:,np.newaxis]])
    except:
        data = info[['id','year','journal','author']].values
    Data = np.vstack([Data,data])
Data_df = pd.DataFrame(Data,columns=[['id','year','journal','author']])


def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)

Save_DataFrame_csv(Data_df,'./Data/node_information_reshape')
