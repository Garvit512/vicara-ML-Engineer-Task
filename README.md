# Activity Recognition from Single Chest-Mounted Accelerometer

The Machine Learning based model able to recognize and predict the human activity recorded by the Chest-Mounted accelerometer with 75% accuracy.

The following steps were performed to achieve that much accuracy.

## Data Collection

The data includes 7 different activities performed by 15 people, the following steps were performed for this:-

1- Merging all 15 files into one and created a single data frame as a single data source.

## EDA & Data Preprocessing

After garnishing data into the desired format, here comes the EDA part which includes the following points:-

1- Handle Missing values \
2- Looking for duplicates \
3- Looking for outliers \
4- Data Normalization and Scaling

## Feature Engineering

Performed feature engineering including Polynomial Features result checked against best model.

## Model Selection

As a cold start, various models were tried which are as follows:-

Machine Learning based:-

1- Logistic Regression \
2- Decision Tree Classifier \
3- Random Forest classifier \
4- XGBoost classifier \
5- XGBRF classifier

Deep Learning based:-

1- MLP \
2- CNN \
3- LSTM

## Model Building & evaluation

Here comes the core part in which _**XGBoost classifier**_ was selected which outperformed all other models during the model selection part.

The hyperparameter tuning was also performed which enhanced the accuracy by 1.5%

## How to run

For model training run the following command:

```
python3 model.py
```

For prediction on custom input (follow the input format)

```
python3 predict.py 2002,2397,1982
```

## Quick FAQ

1- Which model is used? \
XGBoost classifier

2- What is the accuracy? \
75%

3- Why XGBoost? \
On multiple trials with other models, it's outperforming the others

4- Why not Deep Learning based models? \
MLP, CNN, and LSTM were tried but XGBoost was giving good results compared to those.
