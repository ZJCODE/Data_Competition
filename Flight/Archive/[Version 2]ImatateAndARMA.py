# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:05:13 2016

@author: ZJun
"""

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
import statsmodels.api as sm  
from scipy import stats



Path1 = './Data/WIFITAPTag_Mean_All.csv'
WIFITAPTag_Mean_All = pd.read_csv(Path1,parse_dates=['Time'])

Path2 = './Data/WIFIAPTag_List.csv'
WIFIAPTag_List = pd.read_csv(Path2)

WIFIAPTag_List = list(WIFIAPTag_List.WIFIAPTag_List)

   
def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)   
   


def Draw_ACF_PACF(ts,acflag = 100 , pacflag = 100):    
    import statsmodels.api as sm  
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=acflag, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=pacflag, ax=ax2)
 
 
    
def GetTimeSeries(WIFIAPTag):
    Tag_Data = WIFITAPTag_Mean_All[WIFITAPTag_Mean_All.WIFIAPTag == WIFIAPTag]
    Tag_Time_Series = pd.Series(Tag_Data.PassengerCountMean.values , index = Tag_Data.Time)
    return Tag_Time_Series

    
# p=3,q=2

def Do_ARMA(WIFIAPTag,p,q,Draw = False):
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)
    # ARMA model 
    from statsmodels.tsa.arima_model import ARMA
    arma_mod = ARMA(Tag_Time_Series,(p,q)).fit()
    Predict = arma_mod.predict(start='2016-9-14 15:0:0',end='2016-9-14 17:50:0')
    if Draw == True:
        plt.rc('figure', figsize=(12, 8))        
        plt.plot(arma_mod.fittedvalues,'r')
        plt.plot(Tag_Time_Series)
        plt.plot(Predict,'g-')
    return Predict
  
    
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

'''
Tag_E = [tag for tag in WIFIAPTag_List if tag.startswith('E')]
Tag_T = [tag for tag in WIFIAPTag_List if tag.startswith('T')]
Tag_W = [tag for tag in WIFIAPTag_List if tag.startswith('W')]

WIFIAPTag = Tag_T[0]
'''

    
def Compard_Predict(WIFIAPTag,Draw = False):
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)
    try:
        ARMA = Do_ARMA(WIFIAPTag,3,2)
        Imitate = Do_Imitate(WIFIAPTag)
        Combine = (0.55*Imitate+0.45*ARMA)
    except:
        Combine = Do_Imitate(WIFIAPTag)

    if Draw == True:        
        plt.rc('figure', figsize=(15,10))
        plt.plot(Tag_Time_Series,'k')
        plt.plot(ARMA,'r-')
        plt.plot(Imitate,'g-')
        plt.plot(Combine,'b')
    return Combine




def Predict():
    count=0
    tag = WIFIAPTag_List[0]
    
    Predict = Compard_Predict(tag)
    
    def TransTime(time):
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
    
    for tag in WIFIAPTag_List[1:]:
        Predict = Compard_Predict(tag)
        slice10min = [TransTime(time) for time in Predict.index]
        passengerCount = Predict.values
        WIFIAPTag = [tag]*len(Predict)
        Predict_Result_Part = pd.DataFrame({'passengerCount':passengerCount,'WIFIAPTag':WIFIAPTag,'slice10min':slice10min})
        Predict_Result_Part = Predict_Result_Part[['passengerCount','WIFIAPTag','slice10min']]
        Predict_Result = pd.concat([Predict_Result,Predict_Result_Part])
        count += 1
        print count
        
        
    Path_Result = './Data/airport_gz_passenger_predict.csv'
    
    Save_DataFrame_csv(Predict_Result,Path_Result)
       
       

if __name__ == '__main__':
	Predict()


