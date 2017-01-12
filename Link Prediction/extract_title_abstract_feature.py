import pandas as pd
import numpy as np
from gensim import corpora, models, similarities
from collections import Counter
import nltk

stpwds = set(nltk.corpus.stopwords.words("english"))

node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

titles = list(node_information.title.values)
ids = list(node_information.id.values)
abstracts = list(node_information.abstract.values)

ids_df = pd.DataFrame(node_information.id.values,columns=['id'])

title_num_topics = 150
abstract_num_topics = 150

# Title

#titles = [[w for w in t.lower().split() if (w not in stpwds) and (w.isalpha())] for t in titles]
titles = [[w for w in t.lower().split() if (w not in stpwds) ] for t in titles]
dictionary_title = corpora.Dictionary(titles)
corpus_title = [dictionary_title.doc2bow(text) for text in titles]
tfidf_title = models.TfidfModel(corpus_title)
corpus_tfidf_title = tfidf_title[corpus_title]
lsi = models.LsiModel(corpus_tfidf_title, id2word=dictionary_title, num_topics=title_num_topics) # initialize an LSI transformation
corpus_lsi_title = lsi[corpus_tfidf_title] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

title_feature = ['title_feature_'+str(i) for i in range(title_num_topics)]
title_vector = pd.DataFrame([[i[1] for i in a] for a in list(corpus_lsi_title)],columns = title_feature)

# dict
#id_title_vector_dict = dict(zip(ids,[[i[1] for i in a] for a in list(corpus_lsi_title)]))


# Abstract

#abstracts = [[w for w in t.lower().split() if (w not in stpwds) and (w.isalpha())] for t in abstracts]
abstracts = [[w for w in t.lower().split() if (w not in stpwds) ] for t in abstracts]
dictionary_abstract = corpora.Dictionary(abstracts)
corpus_abstract = [dictionary_abstract.doc2bow(text) for text in abstracts]
tfidf_abstract = models.TfidfModel(corpus_abstract)
corpus_tfidf_abstract = tfidf_abstract[corpus_abstract]
lsi = models.LsiModel(corpus_tfidf_abstract, id2word=dictionary_abstract, num_topics=abstract_num_topics) # initialize an LSI transformation
corpus_lsi_abstract = lsi[corpus_tfidf_abstract] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

abstract_feature = ['title_fea111ture_'+str(i) for i in range(abstract_num_topics)]
abstract_vector = pd.DataFrame([[i[1] for i in a] for a in list(corpus_lsi_abstract)],columns = abstract_feature)

# dict
#id_abstract_vector_dict = dict(zip(ids,list(corpus_lsi_abstract)))


def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()

#Save_Obj(id_title_vector_dict,'./Data/id_title_vector_dict')
#Save_Obj(id_abstract_vector_dict,'./Data/id_abstract_vector_dict')



# Combine

title_feature = pd.concat([ids_df,title_vector],axis = 1)
abstract_feature = pd.concat([ids_df,abstract_vector],axis = 1)

def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)

Save_DataFrame_csv(title_feature,'./Data/title_feature')
Save_DataFrame_csv(abstract_feature,'./Data/abstract_feature')

