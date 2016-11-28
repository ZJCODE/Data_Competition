# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:16:00 2016

@author: ZJun
"""



import pandas as pd
import numpy as np
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


def Imitate1(WIFIAPTag,TrainTime,PredictTime):
    
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
        
        RatioList = Ratio(MeanPredictDayTrainTs/MeanValidTrainTimeTsList)
        PredictTs = np.dot(ValidPredictTimeTsList.T,RatioList)
        
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    TS_Predict = pd.Series(PredictTs,index = PredictTimeRange)
    
    return TS_Predict    
    
def Imitate2(WIFIAPTag,TrainTime,PredictTime):
    
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
        PredictTs = np.dot(ValidPredictTimeTsList.T,RatioList) / len(RatioList)
        
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    TS_Predict = pd.Series(PredictTs,index = PredictTimeRange)
    
    return TS_Predict
    
    
    
#testid = [3,87,234，126,90]

def Do_ARMA(WIFIAPTag,TrainTime,PredictTime,p,q,Draw = False):
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)
    ARMA_Time = [PredictTime[0]-timedelta(2),PredictTime[0] - timedelta(0,0,0,0,10,0)]
    #ARMA_Time = [pd.datetime(2016,9,11,6,0,0),pd.datetime(2016,9,14,15,0,0)]
    Tag_Time_Series = Get_Part_of_TimeSeries(Tag_Time_Series,ARMA_Time)
    # ARMA model 
    from statsmodels.tsa.arima_model import ARMA
    arma_mod = ARMA(Tag_Time_Series,(p,q)).fit()
    Predict = arma_mod.predict(start=str(PredictTime[0]),end=str(PredictTime[1]))
    if Draw == True:
        plt.rc('figure', figsize=(12, 8))        
        plt.plot(arma_mod.fittedvalues,'r')
        plt.plot(Tag_Time_Series)
        plt.plot(Predict,'g-')
    return Predict





def ErrorAnalysis(i,day):
    Est_TrainTime = np.array([pd.datetime(2016,9,day,6,0,0),pd.datetime(2016,9,day,14,50,0)])
    Est_PredictTime = np.array([pd.datetime(2016,9,day,15,0,0),pd.datetime(2016,9,day,17,50,0)])
    y = Get_Part_of_TimeSeries(GetTimeSeries(WIFIAPTag_List[i]),Est_PredictTime)
    prey0 =Imitate1(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime)
    prey1 =Imitate2(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime)
    
    def error(a,b):
        return sum([n*n for n in (a-b)])
        
    imitate1_error = error(prey0,y)
    imitate2_error = error(prey1,y)  # sometimes y is empty [expection]
    
    if np.isnan(imitate1_error):
        imitate1_error = 1
    if np.isnan(imitate2_error):
        imitate2_error = 1
    
    try:        
        prey2=Do_ARMA(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime,4,2)
        arma_error = error(prey2,y)
    except:
        arma_error = 10000000
        
    if np.isnan(arma_error):
        arma_error = 10000000
    '''
    y.plot()
    prey0.plot()
    prey1.plot()
    prey2.plot()
    plt.legend(['real','imitate1','imitate2','arma'])
    title = '2016-9-'+str(day)
    plt.title(title)
    '''
    Error_list = [imitate1_error,imitate2_error,arma_error]
    return Error_list
    

def GetRatio():
    
    import time
       
    Error_Analysis = []
    for i in range(len(WIFIAPTag_List)):
        t1 = time.time()
        Error = np.array([0,0,0])
        for j in range(1,8):
            try:
                e = ErrorAnalysis(i,25-j)
                Error = np.c_[Error,e]
            except:
                print 'Error Com'
        ratio = Ratio(1.0/Error.mean(axis=1))
        Error_Analysis.append(ratio)
        t2 = time.time()
        print '===== Got '+str(i)+'th Ratio base on error analysis=====Cost '+str(t2-t1)+' Seconds==='
            
           
    def Save_Obj(Obj,File_Name):    
        import pickle
        File = File_Name + '.pkl'
        output = open(File, 'wb')
        pickle.dump(Obj, output)
        output.close()
        
    Ratio_Dict = dict(zip(WIFIAPTag_List,Error_Analysis))
    Save_Obj(Ratio_Dict,'Ratio_Dict')
    return Ratio_Dict

def Combine(WIFIAPTag,TrainTime,PredictTime,Ratio_Dict):

    imitate1 = Imitate1(WIFIAPTag,TrainTime,PredictTime)
    imiatte2 = Imitate2(WIFIAPTag,TrainTime,PredictTime)
    try:        
        arma = Do_ARMA(WIFIAPTag,TrainTime,PredictTime,4,2)
        if np.isnan(arma.values).any():
            print 'Nan in ARMA'
            arma = Imitate2(WIFIAPTag,TrainTime,PredictTime)
    except:
        print 'ARMA Failed'
        arma = Imitate2(WIFIAPTag,TrainTime,PredictTime)
        
    ratio = Ratio_Dict[WIFIAPTag]
    Predict = imitate1 * ratio[0] + imiatte2 * ratio[1] + arma * ratio[2]
    
    return Predict
    
    


def Compare(i,day,Ratio_Dict):
    Est_TrainTime = np.array([pd.datetime(2016,9,day,6,0,0),pd.datetime(2016,9,day,14,50,0)])
    Est_PredictTime = np.array([pd.datetime(2016,9,day,15,0,0),pd.datetime(2016,9,day,17,50,0)])
    y = Get_Part_of_TimeSeries(GetTimeSeries(WIFIAPTag_List[i]),Est_PredictTime)
    prey0 =Imitate1(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime)
    prey1 =Imitate2(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime)
    prey2=Do_ARMA(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime,4,2)
    prey3 = Combine(WIFIAPTag_List[i],Est_TrainTime,Est_PredictTime,Ratio_Dict)
    y.plot()
    prey0.plot()
    prey1.plot()
    prey2.plot()
    prey3.plot()
    plt.legend(['real','imitate1','imitate2','arma','combine'])
    title = '2016-9-'+str(day)
    plt.title(title)



def Predict(TrainTime,PredictTime,Ratio_Dict):
    count=0
    tag = WIFIAPTag_List[0]

    Predict = Combine(tag,TrainTime,PredictTime,Ratio_Dict)
    
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
        Predict = Combine(tag,TrainTime,PredictTime,Ratio_Dict)
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

'''
if __name__ == '__main__':
    #Ratio_Dict = GetRatio()
    TrainTime = np.array([pd.datetime(2016,9,25,6,0,0),pd.datetime(2016,9,25,14,50,0)])
    PredictTime = np.array([pd.datetime(2016,9,25,15,0,0),pd.datetime(2016,9,25,17,50,0)])
    Predict(TrainTime,PredictTime,Ratio_Dict)
    
'''

