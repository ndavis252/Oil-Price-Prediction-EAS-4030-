# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:26:44 2024

@author: NDavis
"""

import torch 
import torch.nn as nn
import torch.utils.data as utils
from torch.nn.functional import relu

import pandas as pd

from alpha_vantage.foreignexchange import ForeignExchange

from sklearn.preprocessing import StandardScaler

import numpy as np
import requests
import time
import math

key = 'placeholder'

assert key != 'placeholder' "need to input API key for the file to run"

def get_close_vals(forex_data, symbol):
    dates = forex_data.keys()
    values = [forex_data[key]["4. close"] for key in forex_data]

    # create the numpy array using np.fromiter()
    #arr = np.fromiter(zip(dates, values), dtype=[('date', 'datetime64[D]'), (symbol, float)])
    #arr['date'] = np.array(dates, dtype='datetime64[D]')
    #arr['value'] = np.array(values)
    
    df = pd.DataFrame(dates, columns = ['date'])
    df['value'] = values
    
    return df

def get_forex_data(symbol):
    fx = ForeignExchange(key=key)
    data, meta_data = fx.get_currency_exchange_daily(from_symbol=symbol, to_symbol='USD',outputsize='full')
    return get_close_vals(data,symbol)
    
def join_on_date(arrays):
    # create a dictionary to store the values for each date
    date_dict = {}
    for array in arrays:
        for date, value in zip(array['date'], array[1]):
            if date not in date_dict:
                date_dict[date] = {}
            date_dict[date][array.dtype.names[1]] = value
    
    # create a list of dates with values in all arrays
    common_dates = sorted(date for date, values in date_dict.items() if len(values) == len(arrays))
    
    # create a numpy array with the common dates and values from each array
    common_values = np.empty((len(common_dates), len(arrays) + 1), dtype=[('date', 'datetime64[D]')] + [(array.dtype.names[1], float) for array in arrays])
    common_values['date'] = np.array(common_dates, dtype='datetime64[D]')
    for i, array in enumerate(arrays):
        for j, date in enumerate(common_dates):
            common_values[array.dtype.names[1]][j] = date_dict[date][array.dtype.names[1]]
    
    return common_values

def request_other_data(item, interval):
    url = 'https://www.alphavantage.co/query?function=' + item + '&interval=' + interval + '&apikey=' + key
    r = requests.get(url)
    data = r.json()
    return data['data']

def dicts_to_df(dicts):
    keys = dicts[0].keys()
    arr = []
    for i, d in enumerate(dicts):
        in_arr = []
        for j, k in enumerate(keys):
            in_arr.append(d[k])
        arr.append(in_arr)
    return pd.DataFrame(arr, columns=keys)

def df_cast_type(df):
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.set_index('date', inplace=True)
    return df.dropna()
    

# Assumes dataframe is sorted with most recent time periods at the top
# Use factor = 1 when it is sorted with oldest at the top
def percent_change(data, periods, factor = -1):
    return ((data / data.shift(periods*factor)) -1) * 100

def month_changes(data, months):
    for m in months:
        data[str(m) + 'period change'] = percent_change(data['value'],m)
    return data.dropna()

def symbol_to_data(item, interval='monthly'):
    return df_cast_type(dicts_to_df(request_other_data(item, interval)))

def add_year_month_columns(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    return df

def add_year_month_day_columns(df):
    df['year'] =df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    return df


def create_datetime_index(df):
    # Combine year, month, and day columns into a single string column
    df['date_str'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    
    # Convert string column to datetime column and set it as the index
    df['datetime'] = pd.to_datetime(df['date_str'])
    df.set_index('datetime', inplace=True)
    
    # Remove the intermediate string column
    df.drop(columns=['date_str'], inplace=True)
    
    return df

def merge_data(y, month_x, day_x):
    # Merge with data only available by month
    for x in month_x:
        y = y.merge(x, how='left', on=['year','month'])
    
    # Restore Datetime index
    y = create_datetime_index(y)
    
    # Merge with daily data
    for df in day_x:
        y = y.merge(df, how='left', left_index=True,right_index=True)
        
    return y.dropna()

def split_standardize_data(df):
    y = df.iloc[:,0].to_numpy().reshape(-1,1)
    X = df.iloc[:,4:].to_numpy()
    
    sx = StandardScaler()
    X = sx.fit_transform(X)
    sy = StandardScaler()
    y = sy.fit_transform(y)
    
    train_split = int(len(y)*0.75)
    
    X_train, X_test = torch.from_numpy(X[:train_split,:]).float(), torch.from_numpy(X[train_split:,:]).float()
    y_train, y_test = torch.from_numpy(y[:train_split,:]).float(), torch.from_numpy(y[train_split:,:]).float()
    return X_train, X_test, y_train, y_test, sx, sy

class OilModel(nn.Module):
    def __init__(self, xfeat, yfeat, p=[60,60,60]):
        super().__init__()
        self.l1 = nn.Parameter(torch.rand(xfeat, p[0]))
        self.layer_list = []
        for i, dim in enumerate(p[:-1]):
            setattr(self, f'layer_{i+1}', nn.Parameter(torch.rand(dim, p[i+1])-0.5))
            self.layer_list.append(getattr(self, f'layer_{i+1}'))
        self.out = nn.Parameter(torch.rand(p[-1], yfeat)-0.5)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = torch.relu(x@self.l1)
        x = self.dropout(x)
        for layer in self.layer_list:
            x = relu(x@layer + 0.01)
            x = self.dropout(x)
        x = x@self.out
        return x
  
forex_symbols = ['AUD','GBP','EUR']
day_deltas = [22, 65, 130, 260] #1-month, 3-month, 6-month, 1-year number of business days
forex_arrays = [df_cast_type(get_forex_data(s)) for s in forex_symbols]
forex_changes = [month_changes(df, day_deltas) for df in forex_arrays]

GAS_df = symbol_to_data('NATURAL_GAS', interval='daily')
GAS_change = month_changes(GAS_df, day_deltas)

WTI_df = symbol_to_data('WTI', interval='daily')
forex_changes.append(GAS_change)
forex_changes.append(WTI_df)

time.sleep(60) #have to wait because of restrictions on API calls

monthly_values = ['COPPER','ALUMINUM','ALL_COMMODITIES','CPI','DURABLES']
df_list = [symbol_to_data(m) for m in monthly_values]
month_deltas = [1, 3, 6, 12]
indicators_changes = [add_year_month_columns(month_changes(df, month_deltas)) for df in df_list]

# what we are trying to forecast (number of business days in advance)
WTI_df_3month_from_date_spot_price = add_year_month_day_columns(WTI_df.shift(periods=65))
WTI_df_22month_from_date_spot_price = add_year_month_day_columns(WTI_df.shift(periods=479))
WTI_df_70month_from_date_spot_price = add_year_month_day_columns(WTI_df.shift(periods=1521))

#WTI_df_22month_from_date_spot_price_dropped = merge_data(WTI_df_22month_from_date_spot_price, indicators_changes, forex_changes)
#WTI_df_70month_from_date_spot_price_dropped = merge_data(WTI_df_70month_from_date_spot_price, indicators_changes, forex_changes)
WTI_df_3month_from_date_spot_price_dropped = merge_data(WTI_df_3month_from_date_spot_price, indicators_changes, forex_changes)

X_train, X_test, y_train, y_test, sx, sy = split_standardize_data(WTI_df_3month_from_date_spot_price_dropped)

# Training Parameters
lr = 0.1
batch_size = 200
shuffle=True
num_epochs = 200
loss = nn.MSELoss()

# Create the model
xfeat = X_train.size(dim=1)
yfeat = y_train.size(dim=1)

predictions = []
for p in range(12):
    model = OilModel(xfeat, yfeat, p=[30,25,20,15,15,10,10,5,5]) #good parameters for 3 month
    #model = OilModel(xfeat, yfeat, p=[20,15,10,10,10,5,5]) #22 month
    #model = OilModel(xfeat, yfeat, p=[10,10,5]) #70 month
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    #opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = utils.TensorDataset(X_train, y_train)
    train_loader = utils.DataLoader(train_dataset, batch_size, shuffle)
    num_batches = math.ceil(X_train.size(dim=0)/batch_size)
    
    # Training loop
    loss_list = []
    test_loss_list = []
    for i in range(num_epochs):
        for xb, yb in train_loader:
            model.zero_grad()
            
            ypred = model.forward(xb)
            r = loss(yb, ypred)
            r.backward()
            opt.step()
            
        epoch_mse = loss(y_train, model.forward(X_train))
        loss_list.append(epoch_mse.detach())
        test_loss = loss(y_test, model.forward(X_test))
        test_loss_list.append(test_loss.detach().numpy())
        if (i+1) % 20 == 0:
            print(f'Epoch {i}: training loss = {epoch_mse.detach()}, test loss = {test_loss}')
    
    today_vals = []
    [today_vals.append(df.iloc[0,:5]) for df in indicators_changes]
    [today_vals.append(df.iloc[0,:]) for df in forex_changes]
    today_vals = [val for sublist in today_vals for val in sublist]
    today_vals = sx.transform(np.array(today_vals).reshape(1,-1))
    
    prediction = model.forward(torch.FloatTensor(today_vals))
    prediction = sy.inverse_transform(prediction.detach().reshape(-1,1))
    predictions.append(prediction)
    print('Prediction:', prediction)
