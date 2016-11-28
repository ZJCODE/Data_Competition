# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:15:07 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt




Path1 = './Data/WIFITAPTag_Mean_All.csv'
WIFITAPTag_Mean_All = pd.read_csv(Path1,parse_dates=['Time'])

Path2 = './Data/WIFIAPTag_List.csv'
WIFIAPTag_List = pd.read_csv(Path2)

   
def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)   
   
   
def GetTimeSeries(WIFIAPTag):
    '''
    Get WIFIAPTag 's Time Series
    '''
    Tag_Data = WIFITAPTag_Mean_All[WIFITAPTag_Mean_All.WIFIAPTag == WIFIAPTag]
    Tag_Time_Series = pd.Series(Tag_Data.PassengerCountMean.values , index = Tag_Data.Time)
    return Tag_Time_Series



def Do_Imitate(WIFIAPTag):
    '''
    Imitate previous days behavior
    '''
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)

    Oneday = timedelta(1)
    Twoday = timedelta(2)
    Threeday = timedelta(3)

    PredictTime = [pd.datetime(2016,9,14,15,0,0),pd.datetime(2016,9,14,17,50,0)]

    # 9/14 Imitate 9/13 | 9/12 | 9/11
    ImitateTime = [pd.datetime(2016,9,14,6,0,0),pd.datetime(2016,9,14,15,0,0)] 

    
    # Average People'amount in imitate time range in previous three days 
    ThreeDaysBefore = (sum(Tag_Time_Series[ImitateTime[0]-Oneday:ImitateTime[1]-Oneday]) 
    + sum(Tag_Time_Series[ImitateTime[0]-Twoday:ImitateTime[1]-Twoday]) + 
    sum(Tag_Time_Series[ImitateTime[0]-Threeday:ImitateTime[1]-Threeday]) )/3

    # Today’s People's amount in imitate time range
    Today = sum(Tag_Time_Series[ImitateTime[0]:ImitateTime[1]])
    
    # Use People‘s amount in different day to defind the ratio
    Ratio = Today / ThreeDaysBefore
    
    # Average Time series in predict time range in previous three days  [to improve can use weight]
    Reference = (Tag_Time_Series[PredictTime[0]-Oneday:PredictTime[1]-Oneday].values 
    + Tag_Time_Series[PredictTime[0]-Twoday:PredictTime[1]-Twoday].values + 
    Tag_Time_Series[PredictTime[0]-Threeday:PredictTime[1]-Threeday].values) / 3
    
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    
    TS_Predict = pd.Series(Reference*Ratio,index = PredictTimeRange)
    return TS_Predict



def Predict():
    
    tag = WIFIAPTag_List.WIFIAPTag_List[0]
    
    Predict = Do_Imitate(tag)
    
    def TransTime(time):
        '''
        Transform time to a specific format 
        '''
        date = str(time.date())
        hour = time.hour
        minute = time.minute
        output = date + '-' + str(hour) + '-' + str(minute / 10)
        return output
    
    slice10min = [TransTime(time) for time in Predict.index]
    passengerCount = Predict.values
    WIFIAPTag = [tag]*len(Predict)
    Predict_Result = pd.DataFrame({'passengerCount':passengerCount,'WIFIAPTag':WIFIAPTag,'slice10min':slice10min})
    Predict_Result = Predict_Result[['passengerCount','WIFIAPTag','slice10min']]
    
    for tag in WIFIAPTag_List.WIFIAPTag_List[1:]:
        Predict = Do_Imitate(tag)
        slice10min = [TransTime(time) for time in Predict.index]
        passengerCount = Predict.values
        WIFIAPTag = [tag]*len(Predict)
        Predict_Result_Part = pd.DataFrame({'passengerCount':passengerCount,'WIFIAPTag':WIFIAPTag,'slice10min':slice10min})
        Predict_Result_Part = Predict_Result_Part[['passengerCount','WIFIAPTag','slice10min']]
        Predict_Result = pd.concat([Predict_Result,Predict_Result_Part])
        
        
    Path_Result = './Data/airport_gz_passenger_predict.csv'
    
    Save_DataFrame_csv(Predict_Result,Path_Result)
        
if __name__ == '__main__':
	Predict()

    