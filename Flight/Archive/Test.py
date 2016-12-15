# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:47:05 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


path1 = './Data/WIFITAPTag_Mean_All.csv'    
path2 = './Data/schedual_time_count.csv'


WIFITAPTag_Mean_All = pd.read_csv(path1,parse_dates=['Time'])
schedual_time_count = pd.read_csv(path2,parse_dates=['Time'])



schedual_time_count['d'] = [date.day for date in schedual_time_count.Time]

schedual_time_count['WithInTime'] = [1 if (a.hour>14 and a.hour<18) else 0 for a in schedual_time_count.Time]
    
WithInTimeschedual_time_count = schedual_time_count[schedual_time_count.WithInTime ==1]
T1 = WithInTimeschedual_time_count.pivot_table(values = 'FlightPassengerCount',index=['Place','d'],aggfunc='sum')

    
    

WIFITAPTag_Mean_All['Place'] = [p[:2] for p in WIFITAPTag_Mean_All.WIFIAPTag]
WIFITAPTag_Mean_All
WIFITAPTag_Mean_All['d'] = [date.day for date in WIFITAPTag_Mean_All.Time]
WIFITAPTag_Mean_All['WithInTime'] = [1 if (a.hour>14 and a.hour<18) else 0 for a in WIFITAPTag_Mean_All.Time]
WithInTimeWIFITAPTag_Mean_All = WIFITAPTag_Mean_All[WIFITAPTag_Mean_All.WithInTime ==1]
T2 = WithInTimeWIFITAPTag_Mean_All.pivot_table(values = 'PassengerCountMean',index=['Place','d'],aggfunc='sum')

