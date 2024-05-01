# Data-Insights-Hub
## [Forecasting Amazon Stock Prices with LSTM Neural Networks: A Deep Learning Approach][https://github.com/cleavestone/Data-Analytics/blob/main/AMAZON_STOCK_.ipynb]

## Overview
This project aims to forecast the closing stock prices of Amazon using LSTM (Long Short-Term Memory) neural networks. The LSTM model is a type of recurrent neural network (RNN) capable of learning long-term dependencies, making it suitable for time series forecasting tasks.

## Dataset
The dataset used in this project was obtained from kaggle and  contains historical stock data of Amazon from 2005 to 2019. It includes various features such as opening price, closing price, volume, dividends, and more.

## Technologies Used
Python
Libraries:
Pandas
NumPy
Matplotlib
Scikit-learn
TensorFlow/Keras

## Project Workflow
1. Data Exploration and Cleaning: The dataset is loaded, explored for missing values, and checked for duplicates. Necessary cleaning steps are performed.
2. Visualizing Daily Closing Stocks: The daily closing stock prices are visualized to understand the trend and patterns.
3. Splitting into Train and Test Sets: The dataset is split into training and testing sets, with 80% for training and 20% for testing.
4. Data Normalization: Min-max scaling is applied to normalize the data, ensuring that all features have the same scale.
5. Converting to Supervised Learning: Time series data is transformed into a supervised learning problem by creating input-output pairs.
6. Building the LSTM Model: An LSTM neural network model is constructed using Keras, consisting of multiple LSTM layers followed by dense layers.
7. Model Training: The model is trained on the training dataset using the Adam optimizer and mean squared error loss function.
8. Model Evaluation: The trained model is evaluated on the test dataset using metrics such as root mean squared error (RMSE).
9. Making Predictions: The model is used to make predictions on the test dataset, and the results are transformed back to the original scale.
10. Visualizing Predictions: The predicted closing stock prices are plotted alongside the actual prices for comparison.
    
## Results
The LSTM model achieves an RMSE of approximately 86.13, indicating its ability to make accurate predictions.
