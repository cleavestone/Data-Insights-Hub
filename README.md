# Data-Insights-Hub
## [Forecasting Amazon Stock Prices with LSTM Neural Networks: A Deep Learning Approach](https://github.com/cleavestone/Data-Analytics/blob/main/AMAZON_STOCK_.ipynb)

## Overview
This project aims to forecast the closing stock prices of Amazon using LSTM (Long Short-Term Memory) neural networks. The LSTM model is a type of recurrent neural network (RNN) capable of learning long-term dependencies, making it suitable for time series forecasting tasks.

## Dataset
The dataset used in this project was obtained from kaggle and  contains historical stock data of Amazon from 2005 to 2019. It includes various features such as opening price, closing price, volume, dividends, and more.

## Technologies Used
Python Libraries:
1. Pandas
2. NumPy
3. Matplotlib
4. Scikit-learn
5. TensorFlow/Keras

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
![](https://github.com/cleavestone/Data-Insights-Hub/blob/main/image_dir/amazon.png)



## [Predicting Customer Churn in Banking](https://github.com/cleavestone/Data-Analytics/blob/main/Credit_Card_Customer_Churn.ipynb)
Introduction
In today's dynamic banking sector, customer retention is paramount. Banks are increasingly seeking proactive solutions to prevent churn and maintain robust customer relationships. Our goal is to forecast customer churn by leveraging a dataset sourced from Kaggle, containing information from 10,000 customers. This dataset includes crucial factors such as age, salary, marital status, credit card limit, and card category. By harnessing the power of machine learning, we aim to provide insights and construct a predictive model to help banks anticipate churn and implement preemptive measures.

## Dependencies
Python Libraries
1. Pandas
2. NumPy
3. Seaborn
4. Matplotlib
   
## Data Exploration
1. Loaded and inspected the dataset.
2. Explored data dimensions and columns.
3. Checked for missing values and duplicates.
4. Conducted descriptive statistics and visualized data distributions.
5. Explored relationships between variables using correlation matrices and visualizations.
   
## Data Preprocessing
1. Converted categorical variables to numerical using Label Encoding and One-Hot Encoding.
2. Performed train-test split to prepare data for modeling.
   
## Building the Model
1. Employed a Random Forest Classifier to predict customer churn.
2. Evaluated model performance using accuracy score and confusion matrix.
3. Identified key features influencing churn using feature importance analysis.
   
## Results and Insights
1. Achieved an accuracy score of approximately 96.15% in predicting customer churn.
2. Key features influencing churn include total transaction amount, total transaction count, and total revolving balance.
3. Provided demographic analysis for churned customers to facilitate targeted retention strategies.
   
## Next Steps
1. Improve model performance through hyperparameter tuning and feature engineering.
2. Explore advanced machine learning techniques such as neural networks for enhanced predictive capabilities.
3. Deploy the model in a production environment for real-time churn prediction.
![](https://github.com/cleavestone/Data-Insights-Hub/blob/main/image_dir/feature_import.png)
Dataset source: [](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/download?datasetVersionNumber=1)


[Loan Eligibility Prediction Project](https://github.com/cleavestone/Data-Analytics/blob/main/Loans%20application%20.ipynb)

## Problem Statement
Dream Housing Finance company wants to automate the loan eligibility process based on customer details such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. The objective is to predict whether a customer is eligible for a loan amount or not.

## Dataset
We utilized a dataset provided by Dream Housing Finance company, containing customer details such as:
Loan_ID
Gender
Married
Dependents
Education
Self_Employed
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
Property_Area
Loan_Status

## Exploratory Data Analysis (EDA)
1. Checked the shape of the dataset
2. Checked column names and data types
3. Identified missing values and handled them
4. Checked for duplicate values
5. Conducted descriptive statistics on numerical variables
6. Visualized the distribution of numerical variables
7. Explored the cardinality of categorical variables
8. Created a correlation matrix to identify relationships between variables
9. Checked for outliers in numerical variables
10. Examined the imbalance in the dataset and handled it by oversampling the majority class
    
## Preprocessing
1. Handled missing values using SimpleImputer
2. Encoded the target variable using LabelEncoder
3. Encoded categorical data using LabelEncoder and one-hot encoding
4. Performed feature scaling using MinMaxScaler
   
## Model Building
We built four different models for this problem:

1. Logistic Regression
2. XGBoost Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
For each model, we trained and evaluated its performance on the test set.

## Model Evaluation
1. Logistic Regression:
Training Accuracy: 71.41%,
Testing Accuracy: 69.82%
2. XGBoost Classifier:
Training Accuracy: 99.26%,
Testing Accuracy: 86.98%
3. Random Forest Classifier:
Training Accuracy: 70.67%
Testing Accuracy: 73.96%
4. Gradient Boosting Classifier:
Training Accuracy: 84.15%
Testing Accuracy: 76.92%

## Cross-Validation
Due to limited data, all models exhibited some degree of overfitting. To obtain a more accurate score for the XGBoost model, we utilized cross-validation technique. The mean accuracy obtained through cross-validation was approximately 88.50%.

## Conclusion
In conclusion, the XGBoost model showed the highest performance among the models tested, with an accuracy of 86.98% on the test set. However, further optimization and fine-tuning of the models could potentially improve their performance.
