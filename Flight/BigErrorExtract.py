# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 12:09:58 2016

@author: ZJun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:10:18 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

path1 = './Data/flight_time_count.csv'
path2 = './Data/checkin_time_count.csv'
path3 = './Data/security_time_count.csv'
path4 = './Data/WIFITAPTag_Mean_All.csv'    

flight_time_count = pd.read_csv(path1,parse_dates=['Time'])
checkin_time_count = pd.read_csv(path2,parse_dates=['Time'])
security_time_count = pd.read_csv(path3,parse_dates=['Time'])

''' Example

   FlightPassengerCount Place                Time
0                   179    E1 2016-09-11 00:10:00
1                   173    W1 2016-09-11 00:10:00
2                    43    W2 2016-09-11 00:10:00
3                     0   NaN 2016-09-11 00:20:00
4                   206    E2 2016-09-11 00:30:00

'''

WIFITAPTag_Mean_All = pd.read_csv(path4,parse_dates=['Time'])

''' Example

   PassengerCountMean                Time         WIFIAPTag
0                16.2 2016-09-10 19:00:00  E1-1A-1<E1-1-01>
1                19.7 2016-09-10 19:10:00  E1-1A-1<E1-1-01>
2                19.7 2016-09-10 19:20:00  E1-1A-1<E1-1-01>
3                20.5 2016-09-10 19:30:00  E1-1A-1<E1-1-01>
4                20.5 2016-09-10 19:40:00  E1-1A-1<E1-1-01>

'''

WIFIAPTag_List = sorted(list(set(WIFITAPTag_Mean_All.WIFIAPTag)))




def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)   

def GetTimeSeries(WIFIAPTag):
    '''
    Get WIFIAPTag 's Time Series
    '''
    Tag_Data = WIFITAPTag_Mean_All[WIFITAPTag_Mean_All.WIFIAPTag == WIFIAPTag]
    Tag_Time_Series = pd.Series(Tag_Data.PassengerCountMean.values , index = Tag_Data.Time)
    return Tag_Time_Series
    
    
    
  
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
    return TS_Shift
    

# Extract Attribute

def Get_Attribute(WIFITAPTag_Mean_All,Time):
    Attribute = []
    Days_Before = Get_Days_Befor(3)
    for tag in WIFIAPTag_List:
        X = []
        ts = GetTimeSeries(tag)
        TS_Shift = Get_TimeSeries_Shift(ts,1,1)
        for day_before in Days_Before:
            for t in TS_Shift:
                X.append(Get_Part_of_TimeSeries(t, Time- day_before).values)
        DFX = np.array(X).T
        Attribute.append(DFX)
    WIFIAPTag_Attribute_Dict = dict(zip(WIFIAPTag_List,Attribute))
    return WIFIAPTag_Attribute_Dict
        
        

def Get_Enhanced_Attribute(flight_time_count,security_time_count,checkin_time_count,Time):
    EnhancedAttribute = []
    Days_Before = Get_Days_Befor(3)
    for tag in WIFIAPTag_List:
        X=[]
        if str.upper(tag[:2]) in ['W1','W2','W3','E1','E2','E3']:                        
            ts_flight = GetTimeSeries_From_Count(flight_time_count,str.upper(tag[:2]))
            TS_Flight_Shift = Get_TimeSeries_Shift(ts_flight,1,1,1)
            for day_before in Days_Before:
                for t in TS_Flight_Shift:
                    X.append(Get_Part_of_TimeSeries(t, Time- day_before).values)
            
            ts_security = GetTimeSeries_From_Count(security_time_count,str.upper(tag[:2]))
            TS_Security_Shift = Get_TimeSeries_Shift(ts_security,1,1,-1)
            for day_before in Days_Before:
                for t in TS_Security_Shift:
                    X.append(Get_Part_of_TimeSeries(t, Time- day_before).values)
            

            ts_checkin = GetTimeSeries_From_Count(checkin_time_count,str.upper(tag[:2]))
            TS_Checkin_Shift = Get_TimeSeries_Shift(ts_checkin,1,1,-1)
            #TS_Checkin_Shift =[ts_checkin]
            for day_before in Days_Before:
                for t in TS_Checkin_Shift:
                    X.append(Get_Part_of_TimeSeries(t, Time- day_before).values)
            DFX = np.array(X).T
        else:
            DFX = np.nan
        EnhancedAttribute.append(DFX)
    WIFIAPTag_Enhanced_Attribute_Dict = dict(zip(WIFIAPTag_List,EnhancedAttribute))
    return WIFIAPTag_Enhanced_Attribute_Dict
            
        
    
def Combine_Attribute(Attribute,Enhanced_Attribute):
    if np.isnan(Enhanced_Attribute).any():
        return Attribute
    else:
        return np.hstack([Attribute,Enhanced_Attribute])
        
        
def Get_Predict(WIFITAPTag_Mean_All,Time):
    Predict = []
    for tag in WIFIAPTag_List:
        ts = GetTimeSeries(tag)
        TS_Predict = Get_Part_of_TimeSeries(ts,Time).values
        Predict.append(TS_Predict)
    Predict_Dict = dict(zip(WIFIAPTag_List,Predict))
    return Predict_Dict

#--------------------------------

TrainTime = np.array([pd.datetime(2016,9,14,6,0,0),pd.datetime(2016,9,14,12,0,0)])

PredictTime = np.array([pd.datetime(2016,9,14,15,0,0),pd.datetime(2016,9,14,17,50,0)])
#PredictTime = np.array([pd.datetime(2016,9,14,12,0,0),pd.datetime(2016,9,14,14,50,0)])

  
Train_Attr = Get_Attribute(WIFITAPTag_Mean_All,TrainTime)
Train_EnhanceAttr = Get_Enhanced_Attribute(flight_time_count,security_time_count,checkin_time_count,TrainTime)
Y = Get_Predict(WIFITAPTag_Mean_All,TrainTime)
Predict_Attr = Get_Attribute(WIFITAPTag_Mean_All,PredictTime)
Predict_EnhanceAttr = Get_Enhanced_Attribute(flight_time_count,security_time_count,checkin_time_count,PredictTime)

#-----LinearModel---------------------------
    
def Do_LinearModel(WIFIAPTag):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    x = Combine_Attribute(Train_Attr[WIFIAPTag],Train_EnhanceAttr[WIFIAPTag])
    #x = Train_Attr[WIFIAPTag]
    x_for_predict = Combine_Attribute(Predict_Attr[WIFIAPTag],Predict_EnhanceAttr[WIFIAPTag])
    y = Y[WIFIAPTag]
    model.fit(x,y)
    y_predict = model.predict(x_for_predict)
    y_predict = [i if i>0 else 0 for i in y_predict]
    PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
    TS_Predict = pd.Series(y_predict,index = PredictTimeRange)
    return TS_Predict

        


#-----ARMA---------------------------------------------------------------------




def Do_ARMA(WIFIAPTag,p,q,Draw = False):
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)
    ARMA_Time = [pd.datetime(2016,9,11,6,0,0),PredictTime[0] - timedelta(0,0,0,0,10,0)]
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
  
#-----Imitate------------------------------------------------------------------
    
def Do_Imitate(WIFIAPTag):
    '''
    Imitate previous days behavior
    '''
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)

    Oneday = timedelta(1)
    Twoday = timedelta(2)
    Threeday = timedelta(3)
    
    # 9/14 Imitate 9/13 | 9/12 | 9/11
    ImitateTime = TrainTime

    
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


#----------


def test(tag):    
    Do_LinearModel(tag).plot()
    Do_ARMA(tag,3,2).plot()
    Do_Imitate(tag).plot()
    plt.legend(['LinearModel','ARMA','Imitate'])
    plt.title(tag)


#-----------------
# p=4,q = 2
# array([ 0.39066594,  0.63824407,  0.01857195])
# array([ 0.79796709,  0.07148181])

# p=3,q=2
#array([ 0.43207532,  0.57696127,  0.02872377 ])
#array([ 0.79796709,  0.07148181,  0.        ])

def Compard_Predict(WIFIAPTag,Draw = False):
    Tag_Time_Series = GetTimeSeries(WIFIAPTag)
    Imitate = Do_Imitate(WIFIAPTag)

    try:
        ARMA = Do_ARMA(WIFIAPTag,4,2)
        Linear = Do_LinearModel(WIFIAPTag)
        Combine = 0.39066594*Imitate + 0.63824407*ARMA + 0.01857195*Linear
        #Combine = 0.5*Imitate + 0.5*ARMA 
    except:
        try:
            Linear = Do_LinearModel(WIFIAPTag)
            Combine = 0.79796709*Imitate + 0.07148181*Linear # 109088
            #Combine = Imitate  # Without Linear 108238
        except:
            Combine = Imitate
        

    if Draw == True:        
        plt.rc('figure', figsize=(15,10))
        plt.plot(Tag_Time_Series,'k')
        plt.plot(ARMA,'r-')
        plt.plot(Imitate,'g-')
        plt.plot(Linear,'y-')
        plt.plot(Combine,'b')
    return Combine


#----------- Most Important  Part --------------------------------------

def One_Predict(WIFIAPTag,which):
    if which == 0:
        PredictTimeRange = pd.date_range(start = PredictTime[0],end = PredictTime[1] ,freq = '10Min')
        P = pd.Series(np.zeros(len(PredictTimeRange)),index = PredictTimeRange)
    if which == 1:        
        P = Do_Imitate(WIFIAPTag)
    if which == 2:            
        try:        
            P = Do_ARMA(WIFIAPTag,4,2)
        except:
            P = Do_Imitate(WIFIAPTag)
    if which == 3:        
        try:        
            P = Do_LinearModel(WIFIAPTag)
        except:
            P = Do_Imitate(WIFIAPTag)
    if which == 4:
            P = Compard_Predict(WIFIAPTag)
    return P



def Get_Real_Tag(WIFIAPTag,TimeRange):
    
    Real = Get_Part_of_TimeSeries(GetTimeSeries(WIFIAPTag),TimeRange)
    
    return Real


def Compare_Error(P,R):
    error = sum([i*i for i in (P-R)])
    return error
    
    
def Compare_Model_error(M1,M2,M3):
    error = Compare_Error(M1,M2)+ Compare_Error(M1,M3)+Compare_Error(M2,M3)
    if np.isnan(error):
        return 1000000000
    else:
        return error

def Get_Error_List():
    EL = []
    for tag in WIFIAPTag_List:
        M1 = One_Predict(tag,1)
        M2 = One_Predict(tag,2)
        M3 = One_Predict(tag,4)
        EL.append(Compare_Model_error(M1,M2,M3))
    return EL
    
   
EL = Get_Error_List()

DFEL = pd.DataFrame(EL)
    
'''
DFEL.quantile(0.9)

Out[18]: 
0    2632.117528
dtype: float64

DFEL.quantile(0.85)
Out[19]: 
0    1387.892255
dtype: float64
'''   

'''
38      E1-3C<E1-3-14>
162   EC-1G-2<EC-1-25>
186   EC-2A-5<EC-2-05>
228   EC-3C-1<EC-1-16>
231   EC-3C-4<EC-1-14>
296   T1-1A-5<T1-1-18>
324   T1-1D-4<T1-1-03>
502   W1-3C-3<W1-3-20>
541  W2-3B-13<W2-3-26>
'''
   
   
Which_Model=[4 if x < 2500 else 2 for x in EL]   
    
#------------------------------------------------------------------------
'''
def Estiamte_Model_Num():
    Which_Model = []
    for tag in WIFIAPTag_List:
        e = 1000000
        Num = 0
        R = Get_Real_Tag(tag,PredictTime)
        for num in range(1,5):
            P = One_Predict(tag,num)
            Error = Compare_Error(P,R)
            if Error < e:
                e = Error
                Num = num
        Which_Model.append(Num)
    return Which_Model
            
#Which_Model = Estiamte_Model_Num()
            
'''

def Predict():
    count=0
    tag = WIFIAPTag_List[0]
    Predict = One_Predict(tag,Which_Model[0])
    
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
    
    for tag,num in zip(WIFIAPTag_List[1:],Which_Model[1:]):
        Predict = One_Predict(tag,num)
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


def error(A,B):
    return sum([t*t for t in (A.passengerCount - B.passengerCount)])


def Get_Real(WIFITAPTag_Mean_All,TimeRange):
    count=0
    tag = WIFIAPTag_List[0]
    
    Predict = Get_Part_of_TimeSeries(GetTimeSeries(tag),TimeRange)
    
    
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
        Predict = Predict = Get_Part_of_TimeSeries(GetTimeSeries(tag),TimeRange)
        passengerCount = Predict.values
        if len(passengerCount) == 0:
            passengerCount = np.zeros(len(slice10min))
        WIFIAPTag = [tag]*len(slice10min)
        Predict_Result_Part = pd.DataFrame({'passengerCount':passengerCount,'WIFIAPTag':WIFIAPTag,'slice10min':slice10min})
        Predict_Result_Part = Predict_Result_Part[['passengerCount','WIFIAPTag','slice10min']]
        Predict_Result = pd.concat([Predict_Result,Predict_Result_Part])
        count += 1
        print count
    #Path_Result = './Data/airport_gz_passenger_predict_real.csv'
    #Save_DataFrame_csv(Predict_Result,Path_Result)
    
    return Predict_Result


'''
if __name__ == '__main__':
    Predict()
'''


'''
    
    
A = Predict()

B = Get_Real(WIFITAPTag_Mean_All,PredictTime)

def Com(A,B,tag):
    a = A[A.WIFIAPTag == tag].passengerCount
    b = B[B.WIFIAPTag == tag].passengerCount
    plt.plot(range(len(a)),a)
    plt.plot(range(len(b)),b)
    plt.title(tag)
    plt.legend(['test','real'])
    
print error(A,B)



#-----------TEST

def One_Predict(WIFIAPTag,which):
    if which == 1:        
        P = Do_Imitate(WIFIAPTag)
    if which == 2:            
        try:        
            P = Do_ARMA(WIFIAPTag,4,2)
        except:
            P = Do_Imitate(WIFIAPTag)
    if which == 3:        
        try:        
            P = Do_LinearModel(WIFIAPTag)
        except:
            P = Do_Imitate(WIFIAPTag)
    return P
    
    
def Result_Predict(which):
    count=0
    tag = WIFIAPTag_List[0]
    Predict = One_Predict(tag,which)
    
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
        Predict = One_Predict(tag,which)
        slice10min = [TransTime(time) for time in Predict.index]
        passengerCount = Predict.values
        WIFIAPTag = [tag]*len(Predict)
        Predict_Result_Part = pd.DataFrame({'passengerCount':passengerCount,'WIFIAPTag':WIFIAPTag,'slice10min':slice10min})
        Predict_Result_Part = Predict_Result_Part[['passengerCount','WIFIAPTag','slice10min']]
        Predict_Result = pd.concat([Predict_Result,Predict_Result_Part])
        count += 1
        print count
        
        
    #Path_Result = './Data/airport_gz_passenger_predict'+ str(which) +'.csv'
    
    #Save_DataFrame_csv(Predict_Result,Path_Result)
    return Predict_Result
    
    
    
def Get_Coef():
    Imitate = Result_Predict(1)
    ARMA = Result_Predict(2)
    Linear = Result_Predict(3)
    Real = Get_Real(WIFITAPTag_Mean_All,PredictTime)
    X = np.array([Imitate.passengerCount.values,ARMA.passengerCount.values,Linear.passengerCount.values,np.ones(len(Linear))]).T
    y = np.array(Real.passengerCount.values)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X,y)
    return model.coef_
    
# array([ 0.39066594,  0.63824407,  0.01857195])
    
'''


'''


Error = []


def Get_Real_Tag(WIFIAPTag,TimeRange):
    
    Real = Get_Part_of_TimeSeries(GetTimeSeries(WIFIAPTag),TimeRange)
    
    return Real

def Compare_Error(P,R):
    error = sum([i*i for i in (P-R)])
    return error

for tag in WIFIAPTag_List:
    P = Compard_Predict(tag)
    R = Get_Real_Tag(tag,PredictTime)
    Error.append(Compare_Error(P,R))


E = pd.DataFrame(Error)



E.describe()
Out[41]: 
                 0
count   744.000000
mean    148.730133
std     348.033785
min       0.000444
25%            NaN
50%            NaN
75%            NaN
max    4196.480811



Flag = np.array([1 if e >(148) else 0 for e in E.values])

DF_WIFIAPTag_List = pd.DataFrame(WIFIAPTag_List)

Big_Error_Tag = DF_WIFIAPTag_List[Flag == 1]

B = [a[:2] for a in Big_Error_Tag.ix[:,0]]

Tag_Error = dict(zip(WIFIAPTag_List,Error))

'''