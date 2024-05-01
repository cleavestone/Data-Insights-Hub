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


## [Loan Eligibility Prediction Project](https://github.com/cleavestone/Data-Analytics/blob/main/Loans%20application%20.ipynb)

## Problem Statement
Dream Housing Finance company wants to automate the loan eligibility process based on customer details such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. The objective is to predict whether a customer is eligible for a loan amount or not.

## Dataset
We utilized a dataset provided by Dream Housing Finance company, containing customer details such as:
Loan_ID,
Gender,
Married,
Dependents,
Education,
Self_Employed,
ApplicantIncome,
CoapplicantIncome,
LoanAmount,
Loan_Amount_Term,
Credit_History,
Property_Area,
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
[Dataset source](https://github.com/cleavestone/Data-Analytics/blob/main/train_ctrUa4K.csv)

## Conclusion
In conclusion, the XGBoost model showed the highest performance among the models tested, with an accuracy of 86.98% on the test set. However, further optimization and fine-tuning of the models could potentially improve their performance.
![](https://github.com/cleavestone/Data-Insights-Hub/blob/main/image_dir/image34.png)


## Sentiment Analysis of FIFA 2022 Tweets
Welcome to this project notebook! In this project, I perform sentiment analysis on tweets related to FIFA 2022. This analysis involves categorizing tweets as positive, negative, or neutral, providing insights into the public's perception and sentiment regarding one of the world's most anticipated sporting events.

## Context
The FIFA 2022 World Cup captured the attention of millions around the globe, and social media platforms like Twitter were flooded with discussions and reactions to the event. This project aims to perform sentiment analysis on a dataset of 22,000 tweets collected from the first day of FIFA 2022. By understanding the public's sentiment toward the event, we can gain insights into the prevailing opinions and emotions associated with one of the most popular sporting events in the world.

## Project Outline
1. Text Cleaning & Preprocessing: We utilize the spaCy library for efficient text cleaning and preprocessing to ensure that the data is in a usable form for analysis.
2. Feature Extraction: GloVe embeddings are employed for robust feature extraction, allowing us to represent the tweet content effectively and capture its semantic meaning.
3. Exploratory Data Analysis: Word frequency analysis is conducted as part of the exploratory data analysis (EDA) phase to identify important terms and patterns within the tweets.
4. Machine Learning Models: Several machine learning models are built and evaluated, including XGBoost, Random Forest Classifier, Multinomial Logistic Regression, and K-Nearest Neighbors (KNN) classification.
5. Hyperparameter Tuning: Grid Search is used to optimize the hyperparameters for the XGBoost model, which achieves the highest performance score among the models tested.
   
## Objectives
1. Sentiment Classification: Accurately categorize tweets as positive, negative, or neutral.
2. Model Performance: Build and compare multiple machine learning models to determine the most effective approach.

![](https://github.com/cleavestone/Data-Insights-Hub/blob/main/image_dir/fifa23.png)

## [Web Scraping BrighterMonday Job Listings](https://github.com/cleavestone/Data-Analytics/blob/main/SCRAPING%20BRIGHTER%20MONDAY%20JOB%20LISTINGS.ipynb) 
This project aims to collect and analyze job listings data from BrighterMonday, a popular job portal in Kenya. By scraping job listings from BrighterMonday, we gain insights into the job market trends, distribution of jobs by industry and location, as well as the types of job opportunities available.

   
## Tools and Libraries Used
1. Python
2. Requests library for making HTTP requests
3. BeautifulSoup library for parsing HTML content
4. Pandas library for data manipulation and analysis
5. Matplotlib library for data visualization
   
## Project Workflow
1. Downloading Web Pages: We dynamically build URLs to scrap multiple pages of job listings from BrighterMonday and download the HTML content of each page.
2. Scraping Data: Using BeautifulSoup, we extract relevant information from the HTML content, including job titles, industries, companies, locations, job types, and salary ranges.
3. Data Cleaning and Exploration: We clean the scraped data by removing duplicates and exploring null values. We also examine unique categories in each column to gain insights into the data.
4. Visualization: We visualize the distribution of jobs by location and industry using bar charts to better understand the trends in the job market.
   
## Data Insights
1. Location Distribution: Most job listings are concentrated in Nairobi, followed by other major cities like Kisumu and Mombasa. Remote job opportunities are also available, with a smaller proportion compared to location-based jobs.
2. Industry Distribution: The sales industry has the highest number of job listings, followed by other sectors such as marketing, human resources, and management & business development.
3. Salary Ranges: The salary ranges vary across different job listings, with some specifying exact figures while others provide a range. This variation reflects the diversity of job opportunities available on BrighterMonday.
4. 
## Future Work
1. Expansion: The project can be expanded to include other job listing websites in Kenya, such as Glassdoor, Indeed, etc., for a more comprehensive analysis of the job market.
2. Advanced Analysis: Advanced analysis techniques, such as sentiment analysis on job descriptions or predicting salary ranges based on job titles and industries, can be implemented to provide deeper insights.
   
## Conclusion
This project demonstrates the process of web scraping job listings data from BrighterMonday and analyzing it to gain insights into the job market trends in Kenya. The findings can be valuable for job seekers, employers, and policymakers to make informed decisions.
