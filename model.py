import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from xgboost import XGBClassifier
import pickle

# Loading data

path = 'data/Activity Recognition from Single Chest-Mounted Accelerometer/'

files = glob.glob(path + "/*.csv")

li = []

for user_no, file in enumerate(files):
    intermediate_df = pd.read_csv(file, index_col=None, header=0, names=['sequential number', 'x acceleration',
                                                                         'y acceleration', 'z acceleration', 'label'])
    intermediate_df['user_num'] = user_no+1
    li.append(intermediate_df)

df = pd.concat(li, axis=0, ignore_index=True)
cols = ['user_num', 'sequential number', 'x acceleration',
        'y acceleration', 'z acceleration', 'label']
df = df[cols]

df = df[df['label'] != 0]
df = df.sample(frac=1)
# df = df.iloc[:round(len(df)*0.5),:]


# Data Splitting
X = df.iloc[:, 2:5]
y = df.iloc[:, 5:6]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, shuffle=True)


# model
xgb_classfier = XGBClassifier(
    booster='gbtree', learning_rate=0.3, n_estimators=300,  max_depth=8, min_child_weight=2)
xgb_classfier = xgb_classfier.fit(X_train, y_train)

# prediction
y_pred = xgb_classfier.predict(X_test)

# saving the model
pickle_out = open("classifier.pkl", mode="wb")
pickle.dump(xgb_classfier, pickle_out)
pickle_out.close()
