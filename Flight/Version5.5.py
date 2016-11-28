# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:41:23 2016

@author: ZJun
"""



import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt


path1 = './Data/WIFITAPTag_Mean_All.csv'    
path2 = './Data/schedual_time_count.csv'


WIFITAPTag_Mean_All = pd.read_csv(path1,parse_dates=['Time'])
schedual_time_count = pd.read_csv(path2,parse_dates=['Time'])



def Normalize(t):
    return (t-np.mean(t))/np.std(t)
    
    
'''
Normalize(GetTimeSeries_From_Count(schedual_time_count,'E1')).shift(0)[-300:].plot()
Normalize(GetTimeSeries('E1-3B<E1-3-29>'))[-200:].plot()
'''


def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)   

WIFIAPTag_List = sorted(list(set(WIFITAPTag_Mean_All.WIFIAPTag)))


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
    
    
def Get_Part_of_TimeSeries(TS,TimeRange):
    '''
    Input [start_time,end_time]
    '''
    return TS[TimeRange[0]:TimeRange[1]]




def GenerateTs_0(Time):
    timerange = pd.date_range(start = Time[0],end = Time[1] ,freq = '10Min')
    ts = pd.Series(np.zeros(len(timerange)),index = timerange)
    return ts
    
    
def TsList(WIFIAPTag,Time):
    ts_list=[]
    ts = GetTimeSeries(WIFIAPTag)
    for i in range(1,15):
        TimeRange = Time - timedelta(i)
        ts_part = Get_Part_of_TimeSeries(ts,TimeRange)
        if len(ts_part) == 0 or ts_part.isnull().any():
            ts_list.append(GenerateTs_0(TimeRange))
        else:
            ts_list.append(ts_part)
    return np.array(ts_list)
            
   
def TrueFalseListCombine(TFlist1,TFlist2):
    return [l1 and l2 for l1,l2 in zip(TFlist1,TFlist2)]
 
def ExceptOutlier(ts_list):
    
    Mean = pd.DataFrame([np.mean(i) for i in ts_list])
    mean_low = Mean > Mean.quantile(0.1)
    mean_up = Mean < Mean.quantile(0.9)
    TF = TrueFalseListCombine(mean_low.values,mean_up.values)
    mean_index = Mean[TF].index.values
    
    Std = pd.DataFrame([np.std(i) for i in ts_list])
    std_low = Std > Std.quantile(0.1)
    std_up = Std < Std.quantile(0.9)
    TF = TrueFalseListCombine(std_low.values,std_up.values)
    std_index = Std[TF].index.values
    
    valid_index = list(set(mean_index)&set(std_index))
    
    return valid_index # i means minues i+1 day
    

    
def DrawTsList(ts_list):
    plt.plot(ts_list.T)
    
def Ratio(L):
    return np.array(L*1.0/sum(L))
    
    
def Imitate(WIFIAPTag,TrainTime,PredictTime):
    
    TrainTimeTsList = TsList(WIFIAPTag,TrainTime)
    PredictTimeTsList = TsList(WIFIAPTag,PredictTime)
    IndexWithoutOutlier = ExceptOutlier(PredictTimeTsList)
    
    ValidTrainTimeTsList = TrainTimeTsList[IndexWithoutOutlier]
    ValidPredictTimeTsList = PredictTimeTsList[IndexWithoutOutlier]
    PredictDayTrainTs = Get_Part_of_TimeSeries(GetTimeSeries(WIFIAPTag),TrainTime)
    
    if len(PredictDayTrainTs) == 0:
        PredictTs = ValidPredictTimeTsList.mean(axis=0)
    else:
        MeanPredictDayTrainTs = PredictDayTrainTs.mean()
        MeanValidTrainTimeTsList = ValidTrainTimeTsList.mean(axis=1)
        
        RatioList = MeanPredictDayTrainTs/MeanValidTrainTimeTsList
        PredictTs = np.dot(ValidPredictTimeTsList.T,RatioList)/len(RatioList)
        
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    TS_Predict = pd.Series(PredictTs,index = PredictTimeRange)
    
    return TS_Predict

# Remove Biggest variance  And Under Mean

#testid = [3,87,234，126,90]


def Predict(TrainTime,PredictTime):
    count=0
    tag = WIFIAPTag_List[0]

    Predict = Imitate(tag,TrainTime,PredictTime)
    
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
        Predict = Imitate(tag,TrainTime,PredictTime)
        slice10min = [TransTime(time) for time in Predict.index]
        passengerCount = Predict.values
        WIFIAPTag = [tag]*len(Predict)
        Predict_Result_Part = pd.DataFrame({'passengerCount':passengerCount,'WIFIAPTag':WIFIAPTag,'slice10min':slice10min})
        Predict_Result_Part = Predict_Result_Part[['passengerCount','WIFIAPTag','slice10min']]
        Predict_Result = pd.concat([Predict_Result,Predict_Result_Part])
        count += 1
        print count
        
        
    Path_Result = './Data/airport_gz_passenger_predict.csv'
    
    Predict_Result['passengerCount'] = np.nan_to_num(Predict_Result.passengerCount)    
    
    Save_DataFrame_csv(Predict_Result,Path_Result)
    return Predict_Result 


if __name__ == '__main__':
    TrainTime = np.array([pd.datetime(2016,9,25,6,0,0),pd.datetime(2016,9,25,14,50,0)])
    PredictTime = np.array([pd.datetime(2016,9,25,15,0,0),pd.datetime(2016,9,25,17,50,0)])
    Predict(TrainTime,PredictTime)
    


def Com(i,day):
    Est_TrainTime = np.array([pd.datetime(2016,9,day,6,0,0),pd.datetime(2016,9,day,14,50,0)])
    Est_PredictTime = np.array([pd.datetime(2016,9,day,15,0,0),pd.datetime(2016,9,day,17,50,0)])
    y = Get_Part_of_TimeSeries(GetTimeSeries(WIFIAPTag_List[i]),Est_PredictTime)
    prey =Imitate(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime)
    y.plot()
    prey.plot()
    plt.legend(['real','pred'])
    title = '2016-9-'+str(day)
    plt.title(title)
