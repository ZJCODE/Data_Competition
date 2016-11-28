# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:00:48 2016

@author: ZJun
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

def Get_Place_Data(WIFITAPTag_Mean_All,Place,floor = False ):
    '''
    Place can equal to 'W' or 'E' or 'W2' or 'T' or 'W2-1' etc.
    '''
    Len = len(Place)
    WIFITAPTag_Mean_All['Loc'] = [Tag[:Len] for Tag in WIFITAPTag_Mean_All.WIFIAPTag]
    Loc_WIFITAPTag_Mean_All = WIFITAPTag_Mean_All[WIFITAPTag_Mean_All.Loc == Place]
    if floor == False:
        return Loc_WIFITAPTag_Mean_All[['PassengerCountMean','Time','WIFIAPTag']]
    else:
        Loc_WIFITAPTag_Mean_All['Floor'] = [tag[3] for tag in Loc_WIFITAPTag_Mean_All.WIFIAPTag]
        Loc_floor_WIFITAPTag_Mean_All = Loc_WIFITAPTag_Mean_All[Loc_WIFITAPTag_Mean_All.Floor == floor]
        return Loc_floor_WIFITAPTag_Mean_All
     

    
def Get_Place_Data_Except_Loc(WIFITAPTag_Mean_All,Place,floor = False ):
    '''
    WIFITAPTag_Mean_All Data Removes Get_Place_Data Part
    '''
    Loc_WIFITAPTag_Mean_All = Get_Place_Data(WIFITAPTag_Mean_All,Place,floor = floor)
    WIFITAPTag_Mean_All = WIFITAPTag_Mean_All[['PassengerCountMean','Time','WIFIAPTag']]
    WIFITAPTag_Mean_All_Except_Loc = WIFITAPTag_Mean_All[~WIFITAPTag_Mean_All.isin(Loc_WIFITAPTag_Mean_All)].dropna()
    return WIFITAPTag_Mean_All_Except_Loc