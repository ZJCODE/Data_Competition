#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 10:38:56 2017

@author: zhangjun
"""

import pandas as pd
import numpy as np
import networkx as nx
import itertools



train = pd.read_table('./Data/training_set.txt',sep=' ',names=['source','target','link'])
test = pd.read_table('./Data/testing_set.txt',sep=' ',names=['source','target'])


def Save_Obj(Obj,File_Name):    
    import pickle
    File = File_Name + '.pkl'
    output = open(File, 'wb')
    pickle.dump(Obj, output)
    output.close()

G = nx.Graph()
G = G.to_directed()
train_link = train[train.link == 1][['source','target']]
G.add_edges_from(train_link.values)

id_k_core_dict = nx.core_number(G)
Save_Obj(id_k_core_dict,'./Data/id_k_core_dict')


'''
def simrank(G, r=0.8, max_iter=100, eps=1e-4):

    nodes = sorted(G.nodes())
    nodes_i = dict(zip(nodes,range(len(nodes))))

    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))

    for i in range(max_iter):
        print 'round: ' +str(i)
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)

            # evaluating the similarity of current iteration nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0: 
                # if a node has no predecessors then setting similarity to zero
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:                    
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))
    return sim
    
    
   
sim_rank = simrank(G)
Save_Obj(sim_rank,'./Data/sim_rank')
'''