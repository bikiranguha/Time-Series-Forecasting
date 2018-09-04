# try neural networks for time series forecasting (previous time step used to predict next timestep)
import numpy as np
import pandas as pd
import urllib
import os
cwd = os.getcwd()
url = "https://goo.gl/WymYzd"
loc = "C:\\Data\\COE.xls"
urllib.urlretrieve(url,loc) # get the contents of the excel file
Excel_file = pd.ExcelFile(loc)

#print Excel_file.sheet_names
spreadsheet = Excel_file.parse('COE data')
#print spreadsheet.info()

data = spreadsheet['COE$'] # this column of the spreadsheet contains the historical COE price
#print data.head()

# fix some dates
spreadsheet.set_value(194,'DATE','2004-02-15')
spreadsheet.set_value(198,'DATE','2004-04-15')
spreadsheet.set_value(202,'DATE','2004-06-15')

#print spreadsheet['DATE'][193:204]

loc = "C:\\Data\\COE.csv"
spreadsheet.to_csv(loc)

# scale the input attributes
x = data
from sklearn import preprocessing

# part where data is scaled
scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # feature to scale the values
x = np.array(x).reshape((len(x),)) # convert to numpy array
x=np.log(x) # preferred by the author, dont know the exact reason behind this
x=x.reshape(-1,1) # makes x a 2D array (required by scaler)
x = scaler.fit_transform(x) # scaled
x = x.reshape(-1)


# Calculating auto-correlation
from statsmodels.tsa.stattools import pacf
x_pacf = pacf(x,nlags=5,method = 'ols') # contains partial auto-correlation for 5 time steps
#print x_pacf

# use nnet_ts to forecast
from nnet_ts import * # neural network built specifically for time series forecasting
count = 0
ahead = 12
pred = []

while (count<ahead):
	end = len(x) - ahead + count
	np.random.seed(2016)
	fit1 = TimeSeriesNnet(hidden_layers = [7,3],activation_functions = ["tanh","tanh"]) # defining the neural net
	fit1.fit(x[0:end],lag=1,epochs=100) 
	out = fit1.predict_ahead(n_ahead=1) # predict one time step ahead
	print ("Obs: ", count+1, "x = ",round(x[count],4), "prediction = ",round(pd.Series(out),4))
	pred.append(out)
	count = count + 1

