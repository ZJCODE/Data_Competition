import pandas as pd
import numpy as np
import networkx as nx
import community


train = pd.read_table('./Data/training_set.txt',sep=' ',names=['source','target','link'])


G = nx.Graph()
train_link = train[train.link == 1][['source','target']]
G.add_edges_from(train_link.values)

def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L

def DrawGraph(G):
    plt.rc('figure' ,figsize = (15,15))
    nx.draw_networkx(G, pos=nx.spring_layout(G), arrows=True, with_labels=False, node_size=1,node_color='r')

def GetCoreSubNetwork(G,start = False,end = False):    
    G_UnDi = G.to_undirected()
    D = nx.degree(G_UnDi)
    SD = Sort_Dict(D)
    if end == False and start == False:
        Sample_Nodes = [a[0] for a in SD[:]]
    else:
        Sample_Nodes = [a[0] for a in SD[start:end]]
    SubG = nx.subgraph(G_UnDi,Sample_Nodes)
    return SubG
    
def CommunityDetection(DG,n,draw = False ,with_arrow = False,with_label = False):
    
    G = DG.to_undirected()
    Community_Nodes_List = []
    if draw == True:        
        plt.rc('figure',figsize=(12,10))
    #first compute the best partition
    partition = community.best_partition(G) # Nodes With Community tag
    from collections import Counter
    Main = [a[0] for a in Sort_Dict(Counter(partition.values()))[:n]] # Top n Community's Tag
    ZipPartition = partition.items()
    SubNodes = [a[0] for a in ZipPartition if a[1] in Main] # NodesList belong to Top Community
    
    if with_arrow == False:        
        SubG = nx.subgraph(G,SubNodes)
    else:
        SubG = nx.subgraph(DG,SubNodes)
        
    #pos = nx.spectral_layout(SubG)
    #pos = nx.spring_layout(SubG)
    #pos = nx.shell_layout(SubG)
    pos = nx.fruchterman_reingold_layout(SubG)
    if draw == True:        
        if with_label == True:        
            nx.draw(SubG,pos,node_size = 1,alpha =0.1,with_labels=True)
    #drawing
    count = -1

    color = ['b','g','r','c','m','y','k','w']
    for com in set(partition.values()) :
        if com in Main:
            count = count + 1
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            Community_Nodes_List += zip(list_nodes,[count]*len(list_nodes))
            if draw == True:
                nx.draw_networkx_nodes(SubG,pos, list_nodes, node_size = 60,
                                        node_color = color[count],alpha =0.4,with_labels=True)
    if draw == True:  
        if with_label == True:
            plt.legend(['','']+range(1,n+1))
        else:        
            plt.legend(range(1,n+1))      
        nx.draw_networkx_edges(SubG,pos,arrows=True,alpha=0.2)
        plt.show()
        
    Nodes_Category = pd.DataFrame(Community_Nodes_List,columns=['Id','category'])
    Edges = SubG.edges()
    return Nodes_Category , Edges

SubG = GetCoreSubNetwork(G)

N,E = CommunityDetection(SubG,5)

def Save_DataFrame_csv(DF,File_Name):
    File = File_Name + '.csv'
    DF.to_csv(File,encoding='utf8',header=True,index = False)

Save_DataFrame_csv(N,'./Data/papers_network_community_category')