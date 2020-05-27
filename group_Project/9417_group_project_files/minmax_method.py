# This python file is used to pre-process the data and make all of them between 0 and 1
import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score
from matplotlib import pyplot as plt

data = pd.read_csv("flourshing.csv")
data = data.drop(['uid'],axis=1).astype(np.float64)
minmax = preprocessing.MinMaxScaler()
processed_data = minmax.fit_transform(data)
df_count = pd.DataFrame(processed_data)
df_count.to_csv("111.csv")
print(df_count)
