import pandas as pd
import numpy as np
import networkx as nx
import community


train = pd.read_table('./Data/training_set.txt',sep=' ',names=['source','target','link'])


G = nx.Graph()
train_link = train[train.link == 1][['source','target']]
G.add_edges_from(train_link.values)

id_degree_dict = G.degree()
id_bc_dict = nx.betweenness_centrality(G)

def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()

Save_Obj(id_degree_dict,'./Data/id_degree_dict')
Save_Obj(id_bc_dict,'./Data/id_bc_dict')