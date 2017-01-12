import pandas as pd
import numpy as np
from collections import Counter

node_information = pd.read_csv('./Data/node_information.csv',names=['id','year','title','author','journal','abstract'])

def get_set(texts):
    return set(texts)

journal_raw_group = node_information.dropna().pivot_table(values = 'journal',index = 'author',aggfunc = get_set).values

journal_raw_group_set = []
for g in journal_raw_group:
    if g not in journal_raw_group_set:
        journal_raw_group_set.append(g)
    else:
        pass

def combine_set(sets,n,depth):
    sets_sort = sorted(sets,key=lambda x : len(x) ,reverse=True)
    big_sets = [sets_sort[0]] # The Biggest set
    
    while True:
        
        for d in range(depth):          
            # First round
            remove = []
            for s in sets_sort:
                if len(s & big_sets[-1]) >= n or len(s - big_sets[-1]) == 0 :
                    big_sets[-1] = big_sets[-1] | s  # combine to the largest set which matchs the condition
                    remove.append(s)
            for r in remove:
                sets_sort.remove(r)
                
        if len(sets_sort) == 0:
            break
            
        big_sets.append(sets_sort[0])  # The biggest set remain        
    return big_sets

journal_raw_group_sort = sorted(journal_raw_group_set,key=lambda x : len(x) ,reverse=True)
journal_group = combine_set(journal_raw_group_sort,3,3)

top_group_journal = list(journal_group[0])

def Save_List(List,Name):
    File = Name + '.txt'
    pd.DataFrame({Name:List}).to_csv(File,encoding='utf8',header=False,index = False)

Save_List(top_group_journal,'./Data/top_group_journal')
