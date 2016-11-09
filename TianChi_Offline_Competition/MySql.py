# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:25:45 2016

@author: ZJun
"""

#Connect to Mysql

import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta


conn= pymysql.connect(user='root', passwd='ZJMysql310')
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS EMPLOYEE")

sql = """CREATE TABLE EMPLOYEE (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )"""

cur.execute(sql)

sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""

cur.execute(sql)

conn.commit()

sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
       
cur.execute(sql)

results = cur.fetchall()

conn.close()