# Forex and Commodity Price Prediction Model
This repository contains a script, Oil Price Prediction.py, for predicting future prices of commodities based on historical forex and commodity data. The model leverages various machine learning techniques using PyTorch and data preprocessing with Pandas.
Oil Predictions.docx contains predictions for three futures prices for which the project was orinally developed and discusses feature selection. The actual close price for May 2023 futures was $77.29 compared to a prediction of $76.63 (3-months in advance). 

## Description
The goal of this project is to forecast commodity prices, specifically the price of West Texas Intermediate (WTI) crude oil, using historical forex data and other economic indicators. The script collects data from Alpha Vantage, preprocesses it, and trains a neural network to make predictions.

## Key Features
* Data Collection: Fetches daily and monthly forex and commodity data from the Alpha Vantage API.
* Data Preprocessing: Cleans and formats data using Pandas, including handling missing values and normalizing data.
* Feature Engineering: Calculates percentage changes over various periods for better predictive features.
* Model Training: Trains a neural network multiple times to average predictions for robustness.
* Configurable Timeframes: Allows selection of different prediction timeframes (3 months, 22 months, 70 months) with corresponding model architectures.
* Model Averaging: Trains 12 models and averages their predictions to reduce variance and improve accuracy.

## Setup and Usage
### Prerequisites
* Python 3.7 or higher
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Requests
* Alpha Vantage API Key

### Installation
1. Clone the repository:
git clone https://github.com/ndavis252/forex-commodity-prediction.git
cd forex-commodity-prediction

2. Install the required packages:
pip install torch pandas numpy scikit-learn requests alpha_vantage

3. Obtain an Alpha Vantage API key from Alpha Vantage and replace the placeholder in the script with your API key. (https://www.alphavantage.co/)

### Usage
1. Set the desired prediction timeframe by modifying the timeframe variable at the beginning of the script. Options are '3_month', '22_month', or '70_month'.
2. Run the script:
python script.py

### Script Overview
* Data Collection: The script fetches historical forex data for AUD, GBP, and EUR, as well as commodity data for natural gas, WTI, and several economic indicators like copper, aluminum, CPI, etc.
* Data Preprocessing: Functions like df_cast_type and month_changes ensure the data is cleaned and percentage changes are calculated.
* Feature Engineering: The add_date_columns and create_datetime_index functions prepare the data for merging and modeling.
* Model Training: The OilModel class defines a neural network architecture. The script trains the model multiple times, averaging the predictions.
* Predictions: After training, the model makes predictions on the latest data, which are then averaged and outputted.

### Example Output
Epoch 20: Training loss = 0.0023, Test loss = 0.0031
Epoch 40: Training loss = 0.0019, Test loss = 0.0028
...
Prediction: [[70.23]]
Prediction: [[71.10]]
...
Average Prediction: [[70.67]]

### Notes
* Ensure you adhere to the rate limits of the Alpha Vantage API by adjusting the script's timing or using a premium key if necessary.
* Experiment with different hyperparameters and model architectures to improve prediction accuracy.
