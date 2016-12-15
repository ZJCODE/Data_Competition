### WIFI_AP_Passenger_Records_chusai_1stround

- 将timeStamp列转换为Python中的时间对象 [Time]

- 从WIFIAPTag中提取WI-FI位置以及楼层数据 [Loc , Floor]

- MAC 地址分解 [MAC1,MAC2,MAC3]

- Before

  ```
  WIFIAPTag,passengerCount,timeStamp,MAC
  E1-1A-1<E1-1-01>,15,2016-09-10-18-55-04,5869.6c54.d040
  E1-1A-2<E1-1-02>,15,2016-09-10-18-55-04,5869.6c54.cfdc
  E1-1A-3<E1-1-03>,38,2016-09-10-18-55-04,5869.6c54.cf0a
  ```

- After

  ```
  Floor,Loc,MAC1,MAC2,MAC3,PassengerCount,Time,WIFIAPTag
  1,E1,5869,6c54,d040,15,2016-09-10 18:55:04,E1-1A-1<E1-1-01>
  1,E1,5869,6c54,cfdc,15,2016-09-10 18:55:04,E1-1A-2<E1-1-02>
  1,E1,5869,6c54,cf0a,38,2016-09-10 18:55:04,E1-1A-3<E1-1-03>
  ```

  ​


### airport_gz_departure_chusai_1stround

- 将flight_time列转换为Python中的时间对象 [flight_time]

- 将checkin_time列转换为Python中的时间对象 [checkin_time]

- 从flightID中提取出航班的类型(航班号前两位) [flight_Type]

- Before

  ```
  passenger_ID2,flight_ID,flight_time,checkin_time
  177075357.0,FM9358,2016/9/11 13:50:00,2016/9/11 11:57:00
  177075371.0,CZ379,2016/9/11 14:00:00,2016/9/11 11:55:00
  177075476.0,SQ851,,2016/9/11 11:57:00
  ```

- After

  ```
  passenger_ID2,flight_ID,flight_time,checkin_time,flight_Type
  177075357.0,FM9358,2016-09-11 13:50:00,2016-09-11 11:57:00,FM
  177075371.0,CZ379,2016-09-11 14:00:00,2016-09-11 11:55:00,CZ
  177075476.0,SQ851,,2016-09-11 11:57:00,SQ
  ```

  ​

### airport_gz_security_check_chusai_1stround

- 将security_time列转换为Python中的时间对象 [security_time]
- 从passengerID中提取出航班号和ID尾号[PassengerID_Head,PassengerID_Tail]
- Before

```
passenger_ID,security_time,flight_ID
H_CZ1321*045*10SEP16,2016-09-10 4:23:21,CZ1321
AQ1055*002*10SEP16,2016-09-10 4:30:03,AQ1055
ZH9613*027*10SEP16,2016-09-10 4:30:08,ZH9613
```

- After

```
passenger_ID,security_time,flight_ID,PassengerID_Head,PassengerID_Tail
H_CZ1321*045*10SEP16,2016-09-10 04:23:21,CZ1321,H_CZ1321,10SEP16
AQ1055*002*10SEP16,2016-09-10 04:30:03,AQ1055,AQ1055,10SEP16
ZH9613*027*10SEP16,2016-09-10 04:30:08,ZH9613,ZH9613,10SEP16
```

​


