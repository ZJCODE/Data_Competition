# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 19:40:57 2016

@author: ZJun
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta


#Index([u'passenger_ID2', u'flight_ID', u'flight_time', u'checkin_time',u'flight_Type'])
Departure = pd.read_csv('./Data/airport_gz_departure_chusai_1stround_processed.csv',parse_dates=['flight_time','checkin_time'])

Security = pd.read_csv('./Data/airport_gz_security_check_chusai_1stround_processed.csv',parse_dates=['security_time'])


Path3 = './Data/airport_gz_gates.csv'
Gate_Place = pd.read_csv(Path3)
Path4 = './Data/airport_gz_flights_chusai_1stround.csv'
Flight_Gate = pd.read_csv(Path4,parse_dates=['scheduled_flt_time'])

Flight_Gate['scheduled_flt_time'] = [time+timedelta(0,0,0,0,480,0) for time in Flight_Gate.scheduled_flt_time]


# The map form Flight to Gate may change in different day

def Save_Obj(Obj,Path):    
    import pickle
    output = open(Path, 'wb')
    pickle.dump(Obj, output)
    output.close()
    
def Import_Obj(Path):    
    import pickle
    pkl_file = open(Path, 'rb')
    return  pickle.load(pkl_file)

def Get_Date_Flight_Place_Dict(Flight_Gate,Gate_Place):
    
    Gate_Place_Dict = dict(zip(Gate_Place.BGATE_ID,Gate_Place.BGATE_AREA))

    Flight_Gate['Date']=[t.date() for t in Flight_Gate.scheduled_flt_time]
    Date = list(set(Flight_Gate.Date))   
    
    Date_Flight_Place_Dict = dict(zip(Date,[[]]*len(Date)))

    
    for date in Date:
        
        Flight_Gate_Date = Flight_Gate[Flight_Gate.Date == date]
        FlightID_Gate_Date_Dict = dict(zip(Flight_Gate_Date.flight_ID,Flight_Gate_Date.BGATE_ID))
        key = FlightID_Gate_Date_Dict.keys()
        value =[]
        for k in key:
            
            try:
                value.append(Gate_Place_Dict[FlightID_Gate_Date_Dict[k]])
            except:
                value.append('NoInfo')
            
            FlightID_Place_Date_Dict = dict(zip(key,value))
            
            Date_Flight_Place_Dict[date] = FlightID_Place_Date_Dict
            
    
    return Date_Flight_Place_Dict
        
    
    

Date_FlightID_Place_Dict = Get_Date_Flight_Place_Dict(Flight_Gate,Gate_Place)


def Add_Place(Data,Date_FlightID_Place_Dict,TimeColumn):
    Data['Date'] = [t.date() for t in Data[TimeColumn]]
    Place = []
    for flightid,date in zip(Data.flight_ID,Data.Date):
        try:
            Place.append(Date_FlightID_Place_Dict[date][flightid])
        except:
            Place.append('NoInfo')
    Data['Place'] = Place
    return Data
            
            



def SplitDataByEach10Min(Data,timecolumn):
    Data = Data.dropna(subset = [timecolumn]) # Drop Nan Data
    Data = Data.sort_values(by = timecolumn)
    Data.index = range(len(Data))
    TimeSetList = list(set(Data[timecolumn]))
    MinTime = min(TimeSetList) 
    MaxTime = max(TimeSetList)

    def Ceil_Time(time):
        '''
        set 2016-9-12 12:43:10 to 2016-9-12 12:40:00 like this
        '''
        from datetime import timedelta

        def Celi_Minute(Minute):
            '''
            Set 45 to 40 \ 34 to 30 like this
            '''
            return int(10*(round(Minute/10)+1))

        try:            
            T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)+ timedelta(0,0,0,0,Celi_Minute(time.minute),0)
        except:
            try:            
                T = pd.datetime(time.year,time.month,time.day,time.hour,0,0)
            except:
                T = time
        return T
    
    #Initial Time Range with Freq = 10Min
    TimeRange = pd.date_range(start = Ceil_Time(MinTime) , end = Ceil_Time(MaxTime) , freq = '10Min') # Freq : 10Min , 5H , ....
        
    TimeSplitData = [] # Used for storing split data according to time slice
    
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
    

def GetCountData(Data,TimeColumn):
    '''
    # Input 'flight_time' or 'checkin_time' 
    '''
    Data_Process = Add_Place(Data,Date_FlightID_Place_Dict,TimeColumn)
    TimeSplitData , TimeRange = SplitDataByEach10Min(Data_Process,TimeColumn)
     
          
    FlightTime_Count = TimeSplitData[0].pivot_table(values = 'flight_ID' , index = 'Place' , aggfunc = 'count')
    SplitTime = [TimeRange[0]]*len(FlightTime_Count)
    
    DF_FlightTime_Count = pd.DataFrame({'Place':FlightTime_Count.index,'FlightPassengerCount':FlightTime_Count.values,'Time':SplitTime})
    
    FlightTime_Count_All = DF_FlightTime_Count
    
    for i in range(1,len(TimeRange)-1):
        FlightTime_Count = TimeSplitData[i].pivot_table(values = 'flight_ID' , index = 'Place' , aggfunc = 'count')
        if len(FlightTime_Count) > 0:
            SplitTime = [TimeRange[i]]*len(FlightTime_Count)   
            DF_FlightTime_Count = pd.DataFrame({'Place':FlightTime_Count.index,'FlightPassengerCount':FlightTime_Count.values,'Time':SplitTime})
        else:
            SplitTime = [TimeRange[i]]
            DF_FlightTime_Count = pd.DataFrame({'Place':[np.nan],'FlightPassengerCount':[0],'Time':SplitTime})
        
        FlightTime_Count_All = pd.concat([FlightTime_Count_All,DF_FlightTime_Count])
    
    return FlightTime_Count_All ,TimeRange
    
def Get_Flight_People_Num():
    Flight_IDs = list(set(Departure.flight_ID))
    
    Flight_people_num = Departure.pivot_table(values='passenger_ID2',index=['flight_ID','flight_time'],aggfunc='count')
    Flight_Num = []
    for Ids in Flight_IDs:
        Num = np.mean(Flight_people_num[Ids].values)
        Flight_Num.append(Num)
        
    Dict_Flightid_num = dict(zip(Flight_IDs,Flight_Num))
    return Dict_Flightid_num

def GetCountData_schedual(Data,TimeColumn):
    
    Dict_Flightid_num = Get_Flight_People_Num()
    flight_people_num = []
    for Id in Data.flight_ID:
        try:            
            num = Dict_Flightid_num[Id]
            if np.isnan(num):
                print Id
                flight_people_num.append(100)
            else:
                flight_people_num.append(num)
        except:
            flight_people_num.append(100)
    Data['flight_people_num'] = flight_people_num
    
    #-----------------
    Data_Process = Add_Place(Data,Date_FlightID_Place_Dict,TimeColumn)
    TimeSplitData , TimeRange = SplitDataByEach10Min(Data_Process,TimeColumn)
     
          
    FlightTime_Count = TimeSplitData[0].pivot_table(values = 'flight_people_num' , index = 'Place' , aggfunc = 'sum')
    SplitTime = [TimeRange[0]]*len(FlightTime_Count)
    
    DF_FlightTime_Count = pd.DataFrame({'Place':FlightTime_Count.index,'FlightPassengerCount':FlightTime_Count.values,'Time':SplitTime})
    
    FlightTime_Count_All = DF_FlightTime_Count
    
    for i in range(1,len(TimeRange)-1):
        FlightTime_Count = TimeSplitData[i].pivot_table(values = 'flight_people_num' , index = 'Place' , aggfunc = 'sum')
        if len(FlightTime_Count) > 0:
            SplitTime = [TimeRange[i]]*len(FlightTime_Count)   
            DF_FlightTime_Count = pd.DataFrame({'Place':FlightTime_Count.index,'FlightPassengerCount':FlightTime_Count.values,'Time':SplitTime})
        else:
            SplitTime = [TimeRange[i]]
            DF_FlightTime_Count = pd.DataFrame({'Place':[np.nan],'FlightPassengerCount':[0],'Time':SplitTime})
        
        FlightTime_Count_All = pd.concat([FlightTime_Count_All,DF_FlightTime_Count])
    
    return FlightTime_Count_All ,TimeRange
    
        
    
def Generate_flight_checkin__security_count():
    
    flight_time_count , flight_time_range = GetCountData(Departure,'flight_time')
    checkin_time_count , checkin_time_range = GetCountData(Departure,'checkin_time')
    security_time_count , security_time_range = GetCountData(Security,'security_time')
    schedual_time_count , schedual_time_range = GetCountData_schedual(Flight_Gate,'scheduled_flt_time')
    
    path1 = './Data/flight_time_count.csv'
    path2 = './Data/checkin_time_count.csv'
    path3 = './Data/security_time_count.csv'
    path4 = './Data/schedual_time_count.csv'

    
    def Save_DataFrame_csv(DF,File_Path):
        DF.to_csv(File_Path,encoding='utf8',header=True,index = False)
    
    Save_DataFrame_csv(flight_time_count,path1)
    Save_DataFrame_csv(checkin_time_count,path2)
    Save_DataFrame_csv(security_time_count,path3)
    Save_DataFrame_csv(schedual_time_count,path4)



if __name__ == '__main__':
    Generate_flight_checkin__security_count()

#flight_time_count[flight_time_count.Palce=='CZ'].plot(x='Time',y = 'FlightPassengerCount')
#checkin_time_count[checkin_time_count.Place=='CZ'].plot(x='Time',y = 'FlightPassengerCount')



'''

#flight_time_count.plot(x='Time',y='FlightCount')
#checkin_time_count.plot(x='Time',y='FlightCount')

def Sort_Dict(Diction):
    L = list(Diction.items())
    Sort_L = sorted(L,key = lambda x:x[1] , reverse= True)
    return Sort_L

Flight_Type_list = [a[0] for a in Sort_Dict(Counter(flight_time_count.Place))]
t = pd.Series([0]*len(flight_time_range),index=flight_time_range)
test = flight_time_count.pivot_table('FlightPassengerCount',index = ['Place','Time'],aggfunc=sum)

a = test[Flight_Type_list[6]]
m =a+t
m = m.fillna(0)
m.plot()

'''