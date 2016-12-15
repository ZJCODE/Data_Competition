# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:16:31 2016

@author: ZJun
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

#Index([u'Floor', u'Loc', u'MAC1', u'MAC2', u'MAC3', u'PassengerCount', u'Time',u'WIFIAPTag'])
WIFI= pd.read_csv('./Data/WIFI_AP_Passenger_Records_chusai_1stround_processed.csv',parse_dates=['Time'])

def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)
    
def Save_List(List,Name):
    File = './Data/' + Name + '.csv'
    pd.DataFrame({Name:List}).to_csv(File,encoding='utf8',header=True,index = False)


def SplitDataByEach10Min(Data,timecolumn):
    Data = Data.dropna(subset = [timecolumn])
    Data = Data.sort_values(by = timecolumn)
    Data.index = range(len(Data))
    TimeSetList = list(set(Data[timecolumn]))
    MinTime = min(TimeSetList)
    MaxTime = max(TimeSetList)

    def Ceil_Time(time):
        from datetime import timedelta
        def Celi_Minute(Minute):
            return int(10*(round(Minute/10)+1))
        try:            
            T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)+ timedelta(0,0,0,0,Celi_Minute(time.minute),0)
        except:
            try:            
                T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)
            except:
                T = time
        return T
    
    TimeRange = pd.date_range(start = Ceil_Time(MinTime) , end = Ceil_Time(MaxTime) , freq = '10Min') # Freq : 10Min , 5H , ....
    
    TimeSplitData = []
    
    Split_Index_After = 0 ; Split_Index_Before = 0 ; TimeRange_Index = 0
    
    while Split_Index_After < len(Data):
        ThresholdTime = TimeRange[TimeRange_Index]
        
        if Data[timecolumn][Split_Index_After] < ThresholdTime:
            Split_Index_After += 1
        else:
            TimeSplitData.append(Data[Split_Index_Before:Split_Index_After]) 
            Split_Index_Before = Split_Index_After
            TimeRange_Index += 1 
            
    return TimeSplitData , TimeRange
    


TimeSplitData , TimeRange = SplitDataByEach10Min(WIFI,'Time')
 
      
WIFITAPTag_Mean = TimeSplitData[0].pivot_table(values = 'PassengerCount' , index = 'WIFIAPTag' , aggfunc = 'mean')
SplitTime = [TimeRange[0]]*len(WIFITAPTag_Mean)

DF_WIFITAPTag_Mean = pd.DataFrame({'WIFIAPTag':WIFITAPTag_Mean.index,'PassengerCountMean':WIFITAPTag_Mean.values,'Time':SplitTime})

# Combine All Data
WIFITAPTag_Mean_All = DF_WIFITAPTag_Mean

for i in range(1,len(TimeRange)-1):
    try:        
        WIFITAPTag_Mean = TimeSplitData[i].pivot_table(values = 'PassengerCount' , index = 'WIFIAPTag' , aggfunc = 'mean')
        SplitTime = [TimeRange[i]]*len(WIFITAPTag_Mean)
        DF_WIFITAPTag_Mean = pd.DataFrame({'WIFIAPTag':WIFITAPTag_Mean.index,'PassengerCountMean':WIFITAPTag_Mean.values,'Time':SplitTime})
        WIFITAPTag_Mean_All = pd.concat([WIFITAPTag_Mean_All,DF_WIFITAPTag_Mean])
    except:
        pass


WIFITAPTag_Mean_All = WIFITAPTag_Mean_All.sort(['WIFIAPTag','Time'])

Path = './Data/WIFITAPTag_Mean_All.csv'
Save_DataFrame_csv(WIFITAPTag_Mean_All,Path)

WIFIAPTag_List = sorted(list(set(WIFI.WIFIAPTag)))

Save_List(WIFIAPTag_List,'WIFIAPTag_List')
