# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:08:54 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path1 = './Data/flight_time_count.csv'
path2 = './Data/checkin_time_count.csv'
path3 = './Data/security_time_count.csv'
path4 = './Data/WIFITAPTag_Mean_All.csv'    
path5 = './Data/schedual_time_count.csv'

flight_time_count = pd.read_csv(path1,parse_dates=['Time'])
checkin_time_count = pd.read_csv(path2,parse_dates=['Time'])
security_time_count = pd.read_csv(path3,parse_dates=['Time'])
WIFITAPTag_Mean_All = pd.read_csv(path4,parse_dates=['Time'])
schedual_time_count = pd.read_csv(path5,parse_dates=['Time'])



def Normalize(t):
    return (t-np.mean(t))/np.std(t)
    
    
'''
Normalize(GetTimeSeries_From_Count(schedual_time_count,'E1')).shift(0)[-300:].plot()
Normalize(GetTimeSeries('E1-3B<E1-3-29>'))[-200:].plot()
'''

WIFIAPTag_List = sorted(list(set(WIFITAPTag_Mean_All.WIFIAPTag)))




def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)   

def GetTimeSeries(WIFIAPTag):
    '''
    Get WIFIAPTag 's Time Series
    '''
    Tag_Data = WIFITAPTag_Mean_All[WIFITAPTag_Mean_All.WIFIAPTag == WIFIAPTag]
    MinTime = min(Tag_Data.Time)
    MaxTime = max(Tag_Data.Time)
    DataTimeRange = pd.date_range(start = MinTime , end = MaxTime , freq = '10Min')
    ts_0 = pd.Series([0]*len(DataTimeRange),index=DataTimeRange)
    ts =pd.Series(Tag_Data.PassengerCountMean.values , index = Tag_Data.Time)
    TS = ts_0+ts
    TS = TS.fillna(0)
    return TS

    
    
  
#----Linear Model--------------------------------------------------------------
  
    
def GetTimeSeries_From_Count(Data,Place):
    MinTime = min(Data.Time)
    MaxTime = max(Data.Time)
    DataTimeRange = pd.date_range(start = MinTime , end = MaxTime , freq = '10Min')
    ts_0 = pd.Series([0]*len(DataTimeRange),index=DataTimeRange)
    Data_Place = Data[Data.Place == Place]
    ts =pd.Series(Data_Place.FlightPassengerCount.values , index = Data_Place.Time)
    TS = ts_0+ts
    TS = TS.fillna(0)
    return TS
    
    


def Get_Part_of_TimeSeries(TS,TimeRange):
    '''
    Input [start_time,end_time]
    '''
    return TS[TimeRange[0]:TimeRange[1]]
    
    
def Get_Days_Befor(n):
    '''
    return a list of timedelta from 1 day to n day
    '''
    Days = []
    for i in range(1,n+1):
        Days.append(timedelta(i))
    return Days
    
def Get_TimeSeries_Shift(TS,n,step=1,direction = 0):
    '''
    Return a list of timeseries 
    time shift base on what we input
    '''
    TS_Shift = [TS]
    if direction == 0:        
        for i in range(1,n+1):
            TS_Shift.append(TS.shift(i*step))
            TS_Shift.append(TS.shift(-i*step))
    if direction == 1:
        for i in range(1,n+1):
            TS_Shift.append(TS.shift(i*step))
    if direction == -1:
        for i in range(1,n+1):
            TS_Shift.append(TS.shift(-i*step))
    if direction == 2:
        TS_Shift = [TS]
    return TS_Shift
    


def GetMostRelateDay(WIFIAPTag):
    def error(t1,t2):
        return sum([i*i for i in (t1-t2) ])
    t = GetTimeSeries(WIFIAPTag)
    ts=Normalize(np.array(Get_Part_of_TimeSeries(t,TrainTime)))
    n = TrainTime[0].day - 11
    for i in Get_Days_Befor(n):
        ts = np.c_[ts,Normalize(Get_Part_of_TimeSeries(t,TrainTime-i).values)]
    ts = ts.T
    ts_diff = np.diff(ts)
    '''
    L = []
    for i in range(1,len(ts_diff)):
        L.append(error(ts_diff[0],ts_diff[i]))
    
    '''
    L = [10000000]
    for i in range(1,len(ts)):
        L.append(error(ts[0],ts[i]))
    
    return [L.index(i) for i in sorted(L)[:5]]

# Extract Attribute

def Get_Attribute(tag,Time):
    TimeRange = pd.date_range(start = Time[0],end = Time[1] ,freq = '10Min')
    day = GetMostRelateDay(tag)
    Days_Before = [timedelta(i) for i in day]
    X=np.ones(len(TimeRange))
    ts = GetTimeSeries(tag)
    TS_Shift = Get_TimeSeries_Shift(ts,1,1,2)
    for day_before in Days_Before:
        for t in TS_Shift:
            X = np.c_[X,Get_Part_of_TimeSeries(t, Time- day_before).values]
    Attribute = X
    return Attribute
              

def Get_Enhanced_Attribute(tag,Time):
    TimeRange = pd.date_range(start = Time[0],end = Time[1] ,freq = '10Min')
    day = GetMostRelateDay(tag)
    Days_Before = [timedelta(i) for i in day]

    X=np.ones(len(TimeRange))
    if str.upper(tag[:2]) in ['W1','W2','W3','E1','E2','E3']:      
        '''
        ts_flight = GetTimeSeries_From_Count(flight_time_count,str.upper(tag[:2]))
        TS_Flight_Shift = Get_TimeSeries_Shift(ts_flight,1,1,1)
        for day_before in Days_Before:
            for t in TS_Flight_Shift:
                X = np.c_[X,Get_Part_of_TimeSeries(t, Time- day_before).values]
        
        ts_security = GetTimeSeries_From_Count(security_time_count,str.upper(tag[:2]))
        TS_Security_Shift = Get_TimeSeries_Shift(ts_security,1,1,-1)
        for day_before in Days_Before:
            for t in TS_Security_Shift:
                X = np.c_[X,Get_Part_of_TimeSeries(t, Time- day_before).values]
        
        
        ts_checkin = GetTimeSeries_From_Count(checkin_time_count,str.upper(tag[:2]))
        TS_Checkin_Shift = Get_TimeSeries_Shift(ts_checkin,1,1,0)
        #TS_Checkin_Shift =[ts_checkin]
        for day_before in Days_Before:
            for t in TS_Checkin_Shift:
                X = np.c_[X,Get_Part_of_TimeSeries(t, Time- day_before).values]
        '''
        ts_schedual = GetTimeSeries_From_Count(schedual_time_count,str.upper(tag[:2]))
        TS_Schedual_Shift = Get_TimeSeries_Shift(ts_schedual,2,1,1)
        for t in TS_Schedual_Shift:
            X = np.c_[X,Get_Part_of_TimeSeries(t, Time).values]       
        
        DFX = X
    else:
        DFX = np.nan
    
    EnhancedAttribute = DFX    
    return EnhancedAttribute
            
        
    
def Combine_Attribute(Attribute,Enhanced_Attribute):
    if np.isnan(Enhanced_Attribute).all():
        return Attribute
    else:
        return np.nan_to_num(np.hstack([Attribute,Enhanced_Attribute[:,1:]]))


def Get_Predict(WIFIAPTag,Time):
    ts = GetTimeSeries(WIFIAPTag)
    TS_Predict = Get_Part_of_TimeSeries(ts,Time).values
    return TS_Predict

    
def Do_LinearModel(WIFIAPTag,TrainTime,PredictTime):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    x = Combine_Attribute(Get_Attribute(WIFIAPTag,TrainTime),Get_Enhanced_Attribute(WIFIAPTag,TrainTime))
    x_for_predict = Combine_Attribute(Get_Attribute(WIFIAPTag,PredictTime),Get_Enhanced_Attribute(WIFIAPTag,PredictTime))
    y = Get_Predict(WIFIAPTag,TrainTime)
    model.fit(x,y)
    y_predict = model.predict(x_for_predict)
    y_predict = [i if i>0 else 0 for i in y_predict]
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    TS_Predict = pd.Series(y_predict,index = PredictTimeRange)
    return TS_Predict

def Get_Real(WIFIAPTag,PredictTime):
    ts = GetTimeSeries(WIFIAPTag)
    ts_p = Get_Part_of_TimeSeries(ts,PredictTime)
    return ts_p
  



def DealWithExceptionTag(WIFIAPTag,TrainTime,PredictTime):

 
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)

    Oneday = timedelta(1)
    Twoday = timedelta(7)
    n = TrainTime[0].day - 11
    Threeday = timedelta(n)

    ImitateTime = TrainTime             
    
    try:
        
        # Average People'amount in imitate time range in previous three days 
        ThreeDaysBefore = (sum(Tag_Time_Series[ImitateTime[0]-Oneday:ImitateTime[1]-Oneday]) 
        + sum(Tag_Time_Series[ImitateTime[0]-Twoday:ImitateTime[1]-Twoday]) + 
        sum(Tag_Time_Series[ImitateTime[0]-Threeday:ImitateTime[1]-Threeday]) )/3
    
        # Today’s People's amount in imitate time range
        Today = sum(Tag_Time_Series[ImitateTime[0]:ImitateTime[1]])
        
                
        # Use People‘s amount in different day to defind the ratio
        if Today ==0:
            Ratio =1
        else:            
            Ratio = Today / ThreeDaysBefore
        
        # Average Time series in predict time range in previous three days  [to improve can use weight]
        Reference = (Tag_Time_Series[PredictTime[0]-Oneday:PredictTime[1]-Oneday].values 
        + Tag_Time_Series[PredictTime[0]-Twoday:PredictTime[1]-Twoday].values + 
        Tag_Time_Series[PredictTime[0]-Threeday:PredictTime[1]-Threeday].values) / 3
        
    except:
        try:
            
                    # Average People'amount in imitate time range in previous three days 
            ThreeDaysBefore = (sum(Tag_Time_Series[ImitateTime[0]-Oneday:ImitateTime[1]-Oneday]) 
            + 
            sum(Tag_Time_Series[ImitateTime[0]-Threeday:ImitateTime[1]-Threeday]) )/2
        
            # Today’s People's amount in imitate time range
            Today = sum(Tag_Time_Series[ImitateTime[0]:ImitateTime[1]])
            
            # Use People‘s amount in different day to defind the ratio
            if Today ==0:
                Ratio =1
            else:            
                Ratio = Today / ThreeDaysBefore
            
            # Average Time series in predict time range in previous three days  [to improve can use weight]
            Reference = (Tag_Time_Series[PredictTime[0]-Oneday:PredictTime[1]-Oneday].values 
            + 
            Tag_Time_Series[PredictTime[0]-Threeday:PredictTime[1]-Threeday].values) / 2
            
        except:
            # Average People'amount in imitate time range in previous three days 
            T = Tag_Time_Series[PredictTime[0]-Threeday:PredictTime[1]-Threeday] 
            Reference = T.values
            Ratio = 1
            
        
            
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    
    TS_Predict = pd.Series(Reference*Ratio,index = PredictTimeRange)
    return TS_Predict


def Do_ARMA(WIFIAPTag,p,q,TrainTime,PredictTime):
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)
    ARMA_Time = [PredictTime[0]-timedelta(3),PredictTime[0] - timedelta(0,0,0,0,10,0)]
    #ARMA_Time = [pd.datetime(2016,9,11,6,0,0),pd.datetime(2016,9,14,15,0,0)]
    Tag_Time_Series = Get_Part_of_TimeSeries(Tag_Time_Series,ARMA_Time)
    # ARMA model 
    from statsmodels.tsa.arima_model import ARMA
    arma_mod = ARMA(Tag_Time_Series,(p,q)).fit()
    Predict = arma_mod.predict(start=str(PredictTime[0]),end=str(PredictTime[1]))
    return Predict
    
    
def Combine(WIFIAPTag,alpha,beta,TrainTime,PredictTime):
    try:
        Line = Do_LinearModel(WIFIAPTag,TrainTime,PredictTime)
        try:            
            ARMA = Do_ARMA(WIFIAPTag,4,2,TrainTime,PredictTime)
            Combine = alpha*Line+beta*ARMA
        except:
            print 'ARMA Falied'
            Combine = Line  
    except:
        try:           
            print 'Linear Failed'
            Line = DealWithExceptionTag(WIFIAPTag,TrainTime,PredictTime)
            ARMA = Do_ARMA(WIFIAPTag,4,2,TrainTime,PredictTime)
            Combine = alpha*Line+beta*ARMA
        except:
            print 'ARMA Failed'
            Combine = DealWithExceptionTag(WIFIAPTag,TrainTime,PredictTime)
    return Combine
            
    
        






def Com(i):
    
    prey =Combine(WIFIAPTag_List[i],0.8,0.2)
    y =Get_Real(WIFIAPTag_List[i],PredictTime)
    plt.plot(range(len(y)),y,'r-')
    plt.plot(range(len(y)),prey,'g-')
    plt.legend(['real','pred'])

def Estimate(WIFIAPTag,Est_TrainTime,Est_PredictTime):
    try:
        Line = Do_LinearModel(WIFIAPTag,Est_TrainTime,Est_PredictTime)
    except:
        Line = DealWithExceptionTag(WIFIAPTag,Est_TrainTime,Est_PredictTime)
        
    try:
        ARMA = Do_ARMA(WIFIAPTag,4,2,Est_TrainTime,Est_PredictTime)
    except:
        ARMA = DealWithExceptionTag(WIFIAPTag,Est_TrainTime,Est_PredictTime)
        
    Real =Get_Real(WIFIAPTag,Est_PredictTime)
    
    
    
    X = np.array([Line.values,ARMA.values]).T
    y = np.array(Real.values)
    try:        
        coef = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    except:
        coef = np.array([1,0])
    '''
    Combine = coef[0]*Line +coef[1]*ARMA    
    
    Line.plot()
    ARMA.plot()
    Real.plot()
    Combine.plot()

    plt.legend(['l','a','r','c','p'])
    '''
    return coef
    
    
def Predict(Est_TrainTime,Est_PredictTime,TrainTime,PredictTime):
    count=0
    tag = WIFIAPTag_List[0]
    #alpha,beta = Estimate(tag,Est_TrainTime,Est_PredictTime)
    alpha =0.9
    beta = 0.1
    Predict = Combine(tag,alpha,beta,TrainTime,PredictTime)
    
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
        #alpha,beta = Estimate(tag,Est_TrainTime,Est_PredictTime)
        Predict = Combine(tag,alpha,beta,TrainTime,PredictTime)
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
    return Predict_Result    
    

if __name__ == '__main__':
    
    Est_TrainTime = np.array([pd.datetime(2016,9,24,6,0,0),pd.datetime(2016,9,24,14,50,0)])
    
    Est_PredictTime = np.array([pd.datetime(2016,9,24,15,0,0),pd.datetime(2016,9,24,17,50,0)])
    
    
    TrainTime = np.array([pd.datetime(2016,9,24,6,0,0),pd.datetime(2016,9,24,14,50,0)])
    
    PredictTime = np.array([pd.datetime(2016,9,24,15,0,0),pd.datetime(2016,9,24,17,50,0)])
        
    Predict(Est_TrainTime,Est_PredictTime,TrainTime,PredictTime)
