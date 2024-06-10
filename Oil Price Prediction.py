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

# Replace 'placeholder' with your actual API key
key = 'placeholder'

assert key != 'placeholder', "Need to input API key for the file to run"

# Timeframe configurations
timeframe = '3_month'  # Options: '3_month', '22_month', '70_month'
model_params = {
    '3_month': [30, 25, 20, 15, 15, 10, 10, 5, 5],
    '22_month': [20, 15, 10, 10, 10, 5, 5],
    '70_month': [10, 10, 5]
}

# Forex symbols and parameters
forex_symbols = ['AUD', 'GBP', 'EUR']
day_deltas = [22, 65, 130, 260]  # 1-month, 3-month, 6-month, 1-year number of business days
monthly_values = ['COPPER', 'ALUMINUM', 'ALL_COMMODITIES', 'CPI', 'DURABLES']
month_deltas = [1, 3, 6, 12]

def get_close_vals(forex_data, symbol):
    dates = forex_data.keys()
    values = [forex_data[date]["4. close"] for date in dates]
    df = pd.DataFrame({'date': dates, 'value': values})
    return df

def get_forex_data(symbol):
    fx = ForeignExchange(key=key)
    data, _ = fx.get_currency_exchange_daily(from_symbol=symbol, to_symbol='USD', outputsize='full')
    return get_close_vals(data, symbol)

def request_other_data(item, interval):
    url = f'https://www.alphavantage.co/query?function={item}&interval={interval}&apikey={key}'
    response = requests.get(url)
    data = response.json()
    return data['data']

def dicts_to_df(dicts):
    return pd.DataFrame(dicts)

def df_cast_type(df):
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.set_index('date', inplace=True)
    return df.dropna()

def percent_change(data, periods, factor=-1):
    return ((data / data.shift(periods * factor)) - 1) * 100

def month_changes(data, months):
    for m in months:
        data[f'{m}period_change'] = percent_change(data['value'], m)
    return data.dropna()

def symbol_to_data(item, interval='monthly'):
    return df_cast_type(dicts_to_df(request_other_data(item, interval)))

def add_date_columns(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    return df

def create_datetime_index(df):
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['year', 'month', 'day'], inplace=True)
    return df

def merge_data(df, monthly_dfs, daily_dfs):
    for monthly_df in monthly_dfs:
        df = df.merge(monthly_df, how='left', on=['year', 'month'])
    
    df = create_datetime_index(df)
    
    for daily_df in daily_dfs:
        df = df.merge(daily_df, how='left', left_index=True, right_index=True)
        
    return df.dropna()

def split_standardize_data(df):
    y = df.iloc[:, 0].values.reshape(-1, 1)
    X = df.iloc[:, 1:].values
    
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    
    train_split = int(len(y) * 0.75)
    
    X_train = torch.FloatTensor(X[:train_split])
    X_test = torch.FloatTensor(X[train_split:])
    y_train = torch.FloatTensor(y[:train_split])
    y_test = torch.FloatTensor(y[train_split:])
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

class OilModel(nn.Module):
    def __init__(self, x_feat, y_feat, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(x_feat, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers.append(nn.Linear(layers[-1], y_feat))
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

# Fetching forex data
forex_dfs = [get_forex_data(symbol) for symbol in forex_symbols]
forex_changes = [month_changes(df, day_deltas) for df in forex_dfs]

# Fetching natural gas and WTI data
gas_df = symbol_to_data('NATURAL_GAS', interval='daily')
gas_change = month_changes(gas_df, day_deltas)
forex_changes.append(gas_change)

wti_df = symbol_to_data('WTI', interval='daily')
forex_changes.append(wti_df)

time.sleep(60) #have to wait because of restrictions on API calls

# Fetching monthly data
monthly_dfs = [symbol_to_data(symbol, interval='monthly') for symbol in monthly_values]
indicators_changes = [add_date_columns(month_changes(df, month_deltas)) for df in monthly_dfs]

# Preparing target data based on the timeframe
if timeframe == '3_month':
    wti_target = add_date_columns(wti_df.shift(periods=65))
elif timeframe == '22_month':
    wti_target = add_date_columns(wti_df.shift(periods=479))
elif timeframe == '70_month':
    wti_target = add_date_columns(wti_df.shift(periods=1521))

wti_target_dropped = merge_data(wti_target, indicators_changes, forex_changes)

# Splitting and standardizing data
X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_standardize_data(wti_target_dropped)

# Training parameters
lr = 0.1
batch_size = 200
shuffle = True
num_epochs = 200
loss_fn = nn.MSELoss()
num_models = 12

# Create the model
x_feat = X_train.size(1)
y_feat = y_train.size(1)
model_layers = model_params[timeframe]

predictions = []
for _ in range(num_models):
    model = OilModel(x_feat, y_feat, layers=model_layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    train_dataset = utils.TensorDataset(X_train, y_train)
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(yb, y_pred)
            loss.backward()
            optimizer.step()
    
    # Making prediction
    model.eval()
    today_vals = []
    for df in indicators_changes + forex_changes:
        today_vals.append(df.iloc[0, 1:])
    
    today_vals = np.concatenate(today_vals).reshape(1, -1)
    today_vals = scaler_X.transform(today_vals)
    
    prediction = model(torch.FloatTensor(today_vals))
    prediction = scaler_y.inverse_transform(prediction.detach().numpy().reshape(-1, 1))
    predictions.append(prediction)
    print('Prediction:', prediction)

# Average predictions
average_prediction = np.mean(predictions, axis=0)
print('Average Prediction:', average_prediction)
