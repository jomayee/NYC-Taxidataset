import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import datetime
from datetime import datetime

input=pd.read_csv("/users/jpanda/ML/NYCTraffic/finalinput.csv")
Pickup_latitude = ((input.Pickup_latitude >= 40.459518) & (input.Pickup_latitude <= 41.175342))
Pickup_longitude = ((input.Pickup_longitude >= -74.361107) & (input.Pickup_longitude <= -71.903083))
dropoff_latitude = ((input.Dropoff_latitude >= 40.459518) & (input.Dropoff_latitude <= 41.175342))
dropoff_longitude = ((input.Dropoff_longitude >= -74.361107) & (input.Dropoff_longitude <= -71.903083))
data = input[Pickup_latitude & Pickup_longitude & dropoff_latitude & dropoff_longitude]
#Filter data only for NYC latitude and longitude

del data['Store_and_fwd_flag']
#Remove the column

del data['RatecodeID']


data=data.drop(data[(data.Trip_distance <=0)].index)
#Trip distance is zero

data=data[data['Passenger_count']>0]
# Filtered rows where passenger count is 0 as its noise.

data[(data.Fare_amount ==0) & (data.Total_amount !=0) & (data.Tip_amount ==0) & (data.Tolls_amount ==0)]
data=data.drop(data[(data.Fare_amount ==0) & (data.Total_amount !=0) & (data.Tip_amount ==0) & (data.Tolls_amount ==0)].index)
#Fare, tips, tolls are 0 but total is not zero. Exclude these rows as its noise


data=data[(data.Pickuptime != data.Dropofftime)]
data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_latitude == data.Dropoff_latitude)]
#pickup and drop locations are same

data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_latitude == data.Dropoff_latitude) & (data.Trip_distance >=2)]
#pickup and drop locations are same but trip distance is more than 2 miles.

data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_latitude == data.Dropoff_latitude) & (data.Total_amount >=5)]
#pickup and drop locations are same but total amount is more than 5$

data=data.drop(data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_latitude == data.Dropoff_latitude) & (data.Fare_amount >=3)].index)
data=data.drop(data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_latitude == data.Dropoff_latitude) & (data.Total_amount >=5)].index)
data=data.drop(data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_latitude == data.Dropoff_latitude) & (data.Tip_amount >=3)].index)

data[data.Total_amount != data. Tip_amount + data.Tolls_amount + data.Fare_amount + data.Extra + data.Mta_tax]
#Majority of the rows total amount is not equal to sum of tip, tolls, fare, extra, tax
#Hence this column is not of big help. so drop it.

del data['Total_amount']
data=data[data.Tip_amount < 0.25*data.Fare_amount]
#Tip amount is more than 40% of the fare amount which is very rare. Data might be incorrect.

del data['Extra']
del data['Mta_tax']
del data['Payment_type']
del data['Year']
del data['VendorID']
del data['Tolls_amount']

data.Pickuptime=pd.to_datetime(data.Pickuptime)
data.Dropofftime=pd.to_datetime(data.Dropofftime)

#Get trips where distance is greater than 100 miles but pickup and dropoff days are same.
input3=data[(data.Trip_distance > 100) & (data.Pickuptime.dt.strftime('%y-%m-%d') == data.Dropofftime.dt.strftime('%y-%m-%d'))]

#We got 200+ rows for the above command. Some trips have thousands of miles but with pickup and drop in same day which is not possible. We will remove such entries.

#According to current traffic conditions, it takes almost 2 hours to travel 100 miles. Get the records where tripdistance is greater than 100 miles and trip time is less than 2 hours.
input4=input3[(input3.Dropofftime - input3.Pickuptime) < '01:00:00']
#We have around 150 such records, remove them.

#Remove all the trips with greater than 200miles distance as outliers.
data=data.drop(input4.index)
data=data.drop(data[data.Trip_distance > 200].index)

data.to_csv("/users/jpanda/ML/NYCTraffic/inputfile.csv", index=False)
##Create grids so that latitude and longitues can be mapped to zones. We took 50 mile radius for each grid , with better resources, we can reduce to 10 mile radius.

lons = np.arange(-74.4, -73, 0.001, dtype=float)
lats = np.arange(40.4, 41.2, 0.001, dtype=float)
longrid1, latgrid1 = np.meshgrid(lons, lats)
manualgrid = grids.BasicGrid(longrid1.flatten(), latgrid1.flatten())
gpis, gridlons, gridlats = manualgrid.get_grid_points()

data['Zones']=manualgrid.find_nearest_gpi(data['Pickup_longitude'], data['Pickup_latitude'])[1]
data=data.round({'Zone' : 0})
data['Zone']=data['Zone'].astype(int)
data['Hour']=data['Pickuptime'].apply(lambda x: x.hour)
data_aggre=data.groupby(['Weekday', 'Zone', 'Hour']).size().reset_index(name='Count')
data_aggre.to_csv("/users/jpanda/ML/NYCTraffic/inputfile_aggre.csv", index=False)

#The above file will be used to do linear regression. 
#To try another model, we will be creating combined feature models where ZoneID, Houroftheday, Dayoftheweek are combined and used.
features = pandas.concat([pandas.get_dummies(data_aggre.Zone, prefix='Zone'), pandas.get_dummies(data_aggre.Hour, prefix='Hour'), pandas.get_dummies(data_aggre.Weekday, prefix='Weekday')], axis=1)
data_aggre1=pandas.concat([data_aggre, features], axis=1)
data_aggre1.to_csv("/users/jpanda/ML/NYCTraffic/inputfile_aggre1.csv", index=False)





