import pandas
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pygeogrids.grids as grids
import numpy as np

year=""
for dir in os.listdir("/users/jpanda/ML/NYCTraffic/"):
	dir1=os.path.join("/users/jpanda/ML/NYCTraffic/", dir)
	year=dir.split("_")[2].split("-")[0]
	input=pandas.read_csv(dir1)
	input=pandas.DataFrame(input)
	if len(input.columns) > 18:
		input=input.drop(input.columns[17], axis=1)
	input.columns=['VendorID','Pickuptime','Dropofftime','Passenger_count','Trip_distance','Pickup_longitude','Pickup_latitude','RatecodeID','Store_and_fwd_flag',
					'Dropoff_longitude','Dropoff_latitude','Payment_type','Fare_amount','Extra','Mta_tax','Tip_amount','Tolls_amount','Total_amount']
	input.Pickuptime=pandas.to_datetime(input.Pickuptime)
	input3=input[input.Pickuptime.dt.strftime('%H:%M:%S').between('07:00:00','10:00:00')]
	input3['Weekday'] = input3['Pickuptime'].apply(lambda x: x.weekday())
	input3 = input3[input3['Weekday'] < 5 ]
	input3['Year']=year
	input3.to_csv("/users/jpanda/ML/NYC/final_{}".format(dir), index=False)


	
	

	


