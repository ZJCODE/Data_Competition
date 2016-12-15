# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:34:35 2016

@author: ZJun
"""

import pandas as pd

def ReturnTimeElement(Date):
    return [int(t) for t in Date.split('-')]

def TransToTime(TimeElement):
    return pd.datetime(*(TimeElement))
    
def GetTime(Date):
    TimeElement = ReturnTimeElement(Date)
    Time = TransToTime(TimeElement)
    return Time

def Save_DataFrame_csv(DF,File_Path):
    DF.to_csv(File_Path,encoding='utf8',header=True,index = False)



def Process_WIFI_Data():  
    WIFI_AP_Passenger_Records_chusai_1stround = pd.read_csv('./Data/WIFI_AP_Passenger_Records_chusai_1stround.csv')
    
    Time = [];Loc = [];Floor = []
    
    for date in WIFI_AP_Passenger_Records_chusai_1stround.timeStamp:
        Time.append(GetTime(date))
            
    
    for WIFI in WIFI_AP_Passenger_Records_chusai_1stround.WIFIAPTag:
        Location = WIFI[:4]
        loc,floor = Location.split('-')
        Loc.append(str.upper(loc))
        Floor.append(floor)
    
    
              
    WIFI_AP_Passenger_Records_chusai_1stround_After_Process = pd.DataFrame({'WIFIAPTag':
        WIFI_AP_Passenger_Records_chusai_1stround.WIFIAPTag,'Loc':Loc,'Floor':Floor,'PassengerCount':
        WIFI_AP_Passenger_Records_chusai_1stround.passengerCount,'Time':Time})
        
    FilePathWIFI = './Data/WIFI_AP_Passenger_Records_chusai_1stround_processed.csv'
    Save_DataFrame_csv(WIFI_AP_Passenger_Records_chusai_1stround_After_Process,FilePathWIFI)


def Process_Departure_Data():
    
    airport_gz_departure_chusai_1stround = pd.read_csv('./Data/airport_gz_departure_chusai_1stround.csv')
    airport_gz_departure_chusai_1stround['flight_time'] = pd.to_datetime(airport_gz_departure_chusai_1stround.flight_time)
    airport_gz_departure_chusai_1stround['checkin_time'] = pd.to_datetime(airport_gz_departure_chusai_1stround.checkin_time)
    airport_gz_departure_chusai_1stround['flight_Type'] = [flightID[:2] for flightID in airport_gz_departure_chusai_1stround.flight_ID]   
    FilePathAirDeparture = './Data/airport_gz_departure_chusai_1stround_processed.csv'
    Save_DataFrame_csv(airport_gz_departure_chusai_1stround,FilePathAirDeparture)


def Process_Security_Data(): 
    
    airport_gz_security_check_chusai_1stround = pd.read_csv('./Data/airport_gz_security_check_chusai_1stround.csv')
    airport_gz_security_check_chusai_1stround['security_time'] = pd.to_datetime(airport_gz_security_check_chusai_1stround.security_time)
    
    
    FilePathAirSecurity = './Data/airport_gz_security_check_chusai_1stround_processed.csv'
    Save_DataFrame_csv(airport_gz_security_check_chusai_1stround,FilePathAirSecurity)    


if __name__ == '__main__':
    Process_WIFI_Data()
    Process_Departure_Data()
    Process_Security_Data()