# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:01:52 2016

@author: ZJun
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

#Index([u'Floor', u'Loc', u'MAC1', u'MAC2', u'MAC3', u'PassengerCount', u'Time',u'WIFIAPTag'])
WIFI= pd.read_csv('./Data/WIFI_AP_Passenger_Records_chusai_1stround_processed.csv',parse_dates=['Time'])
#Index([u'passenger_ID2', u'flight_ID', u'flight_time', u'checkin_time',u'flight_Type'])
Departure = pd.read_csv('./Data/airport_gz_departure_chusai_1stround_processed.csv',parse_dates=['flight_time','checkin_time'])
#Index([u'passenger_ID', u'security_time', u'flight_ID', u'PassengerID_Head',u'PassengerID_Tail'])
Security = pd.read_csv('./Data/airport_gz_security_check_chusai_1stround_processed.csv',parse_dates=['security_time'])


#--------------------------



def PlotCounter(Count,Sort = 1): 
    
    def Sort_Dict_Value(Diction):
        L = list(Diction.items())
        Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
        return Sort_L

    def Sort_Dict_Key(Diction):
        L = list(Diction.items())
        Sort_L = sorted(L,key = lambda x:x[0] , reverse= True)
        return Sort_L
        
    
    plt.rc('figure', figsize=(len(Count), len(Count)/2))
    if Sort == 1: # Sorted by value
        SortCount = Sort_Dict_Value(Count)
        key = [a[0] for a in SortCount]
        value = [a[1] for a in SortCount]
    else: # Sorted by key
        SortCount = Sort_Dict_Key(Count)
        key = [a[0] for a in SortCount]
        value = [a[1] for a in SortCount]
    pos = np.arange(len(key))
    plt.bar(pos,value,color='k',alpha=0.6,linewidth=1)
    plt.xticks(pos+0.4, key)
    
    
    
def PlotPivotTable(Pivot,Sort = 1): 
    
    def Sort_Value(Pivot):
        Sort_Pivot = Pivot.sort_values(ascending=False)
        return Sort_Pivot    
        
    def Sort_Index(Pivot):
        Sort_Pivot = Pivot.sort_index(ascending=False)
        return Sort_Pivot   
    
    plt.rc('figure', figsize=(len(Pivot), len(Pivot)/2))
    if Sort == 1: # Sorted by value
        SortPivot = Sort_Value(Pivot)
        index = SortPivot.index
        value = SortPivot.values
    else: # Sorted by index
        SortPivot = Sort_Index(Pivot)
        index = SortPivot.index
        value = SortPivot.values
    pos = np.arange(len(index))
    plt.bar(pos,value,color='k',alpha=0.6,linewidth=1)
    plt.xticks(pos+0.4, index)    


#--------------------------------


WIFI['WIFIGroup'] = [str.upper(tag[:5]) for tag in WIFI.WIFIAPTag]

WIFI['WIFILocFloor'] = [str.upper(tag[:4]) for tag in WIFI.WIFIAPTag]

'''
WIFILocFloorNum = WIFI.pivot_table(values='PassengerCount',index='WIFILocFloor',aggfunc=sum)

WIFILocNum = WIFI.pivot_table(values='PassengerCount',index='Loc',aggfunc=sum)

PlotPivotTable(WIFILocFloorNum)
'''

#--------------------------------


