import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

top_group_journal = pd.read_csv('./Data/top_group_journal.txt',names=['journal']).journal.values

author_mean_year = node_information.pivot_table(values='year',index='author',aggfunc='mean')
author_paper_num_year = node_information.pivot_table(values='id',index=['author','year'],aggfunc='count')
journal_paper_num_year = node_information.pivot_table(values='id',index=['journal','year'],aggfunc='count')


author_paper_year_mean_list = []
author_paper_num_year_list = []
journal_paper_num_year_list = []
journal_class = []
id_list = []


for i in range(len(node_information)):
	print 'process :' + str(i) + 'th line'
	info = node_information.ix[i,:]
	
	id_list.append(info.id)

	if info.journal in top_group_journal:
		journal_class.append(1)
	else:
		journal_class.append(0)
	try:                        
		author_paper_year_mean_list.append(author_mean_year[info.author])
	except:                           
		author_paper_year_mean_list.append(np.nan)
	try:
		author_paper_num_year_list.append(author_paper_num_year[info.author][info.year])
	except:
		author_paper_num_year_list.append(np.nan)
	try:
		journal_paper_num_year_list.append(journal_paper_num_year[info.journal][info.year])
	except:
		journal_paper_num_year_list.append(np.nan)


author_journal_feature = pd.DataFrame({'id':id_list,
	'author_paper_year_mean':author_paper_year_mean_list,
	'author_paper_num_year':author_paper_num_year_list,
	'journal_paper_num_year':journal_paper_num_year_list,
	'journal_class':journal_class})

author_journal_feature = author_journal_feature[['id','author_paper_year_mean','author_paper_num_year','journal_paper_num_year','journal_class']]

def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)

Save_DataFrame_csv(author_journal_feature,'./Data/author_journal_feature')


imputer = Imputer(strategy="mean") 
imputer.fit(author_journal_feature)
author_journal_feature_without_nan = imputer.transform(author_journal_feature)
author_journal_feature_without_nan = pd.DataFrame(author_journal_feature_without_nan,columns = author_journal_feature.columns)
author_journal_feature_without_nan.id = author_journal_feature_without_nan.id.astype(int)

Save_DataFrame_csv(author_journal_feature_without_nan,'./Data/author_journal_feature_without_nan')


