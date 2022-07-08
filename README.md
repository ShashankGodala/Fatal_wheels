# Fatal_wheels
This is Data analysis project on Australian road deaths.

```
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```
```
# Importing data
df = pd.read_csv('Crash_Data.csv',low_memory=False)
df.head()
```
```
Crash ID	State	Month	Year	Dayweek	Time	Crash Type	Bus Involvement	Heavy Rigid Truck Involvement	Articulated Truck Involvement	...	Age	National Remoteness Areas	SA4 Name 2016	National LGA Name 2017	National Road Type	Christmas Period	Easter Period	Age Group	Day of week	Time of day
0	20212133	Vic	9	2021	Sunday	0:30	Single	NaN	NaN	NaN	...	38	Inner Regional Australia	Melbourne - Outer East	Yarra Ranges (S)	Arterial Road	No	No	26_to_39	Weekend	Night
1	20214022	SA	9	2021	Saturday	23:31	Multiple	No	No	No	...	28	Major Cities of Australia	Adelaide - North	Playford (C)	NaN	No	No	26_to_39	Weekend	Night
```
```
df.shape
```
(52843, 23)

```
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 52843 entries, 0 to 52842
Data columns (total 23 columns):
 #   Column                         Non-Null Count  Dtype 
---  ------                         --------------  ----- 
 0   Crash ID                       52843 non-null  int64 
 1   State                          52843 non-null  object
 2   Month                          52843 non-null  int64 
 3   Year                           52843 non-null  int64 
 4   Dayweek                        52843 non-null  object
 5   Time                           52803 non-null  object
 6   Crash Type                     52843 non-null  object
 7   Bus Involvement                52821 non-null  object
 8   Heavy Rigid Truck Involvement  32328 non-null  object
 9   Articulated Truck Involvement  52821 non-null  object
 10  Speed Limit                    52141 non-null  object
 11  Road User                      52843 non-null  object
 12  Gender                         52816 non-null  object
 13  Age                            52843 non-null  int64 
 14  National Remoteness Areas      6878 non-null   object
 15  SA4 Name 2016                  6892 non-null   object
 16  National LGA Name 2017         6893 non-null   object
 17  National Road Type             6877 non-null   object
 18  Christmas Period               52843 non-null  object
 19  Easter Period                  52843 non-null  object
 20  Age Group                      52753 non-null  object
 21  Day of week                    52843 non-null  object
 22  Time of day                    52843 non-null  object
dtypes: int64(4), object(19)
memory usage: 9.3+ MB
```

## Data Cleaning

```
# columns 14,15,16,17,18,19 having lot of missing values or insignificant information, thus have been removed
df_new = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,20,21,22]]
df_new.head()
```
```
Crash ID	State	Month	Year	Dayweek	Time	Crash Type	Bus Involvement	Heavy Rigid Truck Involvement	Articulated Truck Involvement	Speed Limit	Road User	Gender	Age	Age Group	Day of week	Time of day
0	20212133	Vic	9	2021	Sunday	0:30	Single	NaN	NaN	NaN	NaN	Motorcycle rider	Male	38	26_to_39	Weekend	Night
1	20214022	SA	9	2021	Saturday	23:31	Multiple	No	No	No	110	Pedestrian	Female	28	26_to_39	Weekend	Night
2	20212096	Vic	9	2021	Saturday	23:00	Single	NaN	NaN	NaN	NaN	Passenger	Male	19	17_to_25	Weekend	Night
3	20212145	Vic	9	2021	Saturday	22:25	Single	NaN	NaN	NaN	NaN	Driver	Male	23	17_to_25	Weekend	Night
4	20212075	Vic	9	2021	Saturday	5:15	Single	NaN	NaN	NaN	NaN	Motorcycle rider	Male	46	40_to_64	Weekend	Night
```
```
# Inspecting all the heavy vehicles involvement columns
df_new.iloc[:,7:10].value_counts()
```
```
Bus Involvement  Heavy Rigid Truck Involvement  Articulated Truck Involvement
No               No                             No                               27141
                                                Yes                               3074
                 Yes                            No                                1476
Yes              No                             No                                 498
No               Yes                            Yes                                105
Yes              No                             Yes                                 22
                 Yes                            No                                  12
dtype: int64
```

```
# combining the bus, heavy and articulated involvement columns into single column 
# populating the null values with 0
df_new = df_new.fillna(value = 0)
# converting the boolean values of the 3 columns into 1 and 0 to perform calculation
df_new['Bus Involvement'] = df_new['Bus Involvement'].map({'Yes':1,'No':0,0:0})
df_new['Heavy Rigid Truck Involvement'] = df_new['Heavy Rigid Truck Involvement'].map({'Yes':1,'No':0,0:0})
df_new['Articulated Truck Involvement'] = df_new['Articulated Truck Involvement'].map({'Yes':1,'No':0,0:0})
# new column is 'Yes' if any of the three old columns says 'Yes'
df_new['Heavy vehicle involvement'] = df_new['Bus Involvement']+df_new['Heavy Rigid Truck Involvement']+df_new['Articulated Truck Involvement']
df_new['Heavy vehicle involvement'] = ['Yes' if i>0 else 'No' for i in df_new['Heavy vehicle involvement']]
print(df_new['Heavy vehicle involvement'].value_counts())
```
```
No     45103
Yes     7740
Name: Heavy vehicle involvement, dtype: int64
```
```
# replacing 0 values in time column to a hour and minute format
df_new['Time'] = df_new['Time'].replace(to_replace = 0,value='0:0')
# merging Time,month and year columns to create one date column
df_new['date'] = df_new['Year'].astype(str) + ' ' + df_new['Month'].astype(str) + ' ' + df_new['Time'].astype(str)
# converting the data type into datetime
df_new['date'] = pd.to_datetime(df_new['date'],format = '%Y %m %H:%M')
df_new['date'].head()
```
```
0   2021-09-01 00:30:00
1   2021-09-01 23:31:00
2   2021-09-01 23:00:00
3   2021-09-01 22:25:00
4   2021-09-01 05:15:00
Name: date, dtype: datetime64[ns]
```
```
# Replacing all the irrelavant and insignificant data to none
df_new['Speed Limit'].replace(to_replace = ['-9','5','15','25',0,'0','<40','Unspecified'],value = None,inplace=True)
# cleaning up the gender column
df_new['Gender'].replace([0,'0','Unspecified'],None,inplace=True)
# cleaning up the age group column
df_new['Age Group'].replace([0,'0'],None,inplace = True)
# cleaning up the age column
df_new['Age'].replace([-9,0],None,inplace = True)
print(df_new['Speed Limit'].value_counts())
print(df_new['Gender'].value_counts())
print(df_new['Age Group'].value_counts())
```
```
100    18248
60     13686
80      6119
110     6038
50      2903
70      2538
90      1137
40       346
75       254
130      116
20        27
10        18
30        14
Name: Speed Limit, dtype: int64
Male      37813
Female    15002
Name: Gender, dtype: int64
17_to_25       13771
40_to_64       13415
26_to_39       12364
75_or_older     5110
0_to_16         4080
65_to_74        4013
Name: Age Group, dtype: int64
```
```
# preparing a new dataframe with cleaned and organized data
# for the purpose of this analysis I decided to trim the data to past 30 years (1991 to 2021)
df_clean = df_new[pd.DatetimeIndex(df_new['date']).year > 1991].iloc[:,[0,18,3,4,15,16,1,11,12,13,14,18,6,10,17]]
df_clean.head()
```
```
Crash ID	date	Year	Dayweek	Day of week	Time of day	State	Road User	Gender	Age	Age Group	date	Crash Type	Speed Limit	Heavy vehicle involvement
0	20212133	2021-09-01 00:30:00	2021	Sunday	Weekend	Night	Vic	Motorcycle rider	Male	38	26_to_39	2021-09-01 00:30:00	Single	None	No
1	20214022	2021-09-01 23:31:00	2021	Saturday	Weekend	Night	SA	Pedestrian	Female	28	26_to_39	2021-09-01 23:31:00	Multiple	110	No
2	20212096	2021-09-01 23:00:00	2021	Saturday	Weekend	Night	Vic	Passenger	Male	19	17_to_25	2021-09-01 23:00:00	Single	None	No
3	20212145	2021-09-01 22:25:00	2021	Saturday	Weekend	Night	Vic	Driver	Male	23	17_to_25	2021-09-01 22:25:00	Single	None	No
4	20212075	2021-09-01 05:15:00	2021	Saturday	Weekend	Night	Vic	Motorcycle rider	Male	46	40_to_64	2021-09-01 05:15:00	Single	None	No
```

## Analysis




