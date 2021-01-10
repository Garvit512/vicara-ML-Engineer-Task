import sys
import pickle
import pandas as pd

xgb_model_loaded = pickle.load(open('classifier.pkl', "rb"))

# inp = [2002, 2397, 1982]

list1 = sys.argv[1].split(',')
inp = list(map(int, list1))

a = pd.DataFrame(inp).T
a.columns = ['x acceleration', 'y acceleration', 'z acceleration']
res = xgb_model_loaded.predict(a)
print("predicted class: ", res[0])
