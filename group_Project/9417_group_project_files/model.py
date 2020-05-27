# In this pythone file, we get the model and model Optimization
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore")


# we already store the pre-processing data to csv files
# because each file has different size, so we need to import
# files individually
path1 = "F://9417/feature_label_2/panas_P.csv"
path2 = "F://9417/feature_label_2/panas_N.csv"
path3 = "F://9417/feature_label_2/flourishing.csv"  

data=pd.read_csv(path1)
# seperate the file to feature and label dataframes
feature = data.iloc[:,1:17]
label = data.iloc[:,17:18]

# get the classification report with accuracy score for all fold
def classification_report_with_accuracy_score(y_true, y_pred):
    result = metrics.classification_report(y_true, y_pred)
    return metrics.accuracy_score(y_true, y_pred)
# --------------------get the initial accuracy with initial feature---------------------- #
def classification(model,feature,label):
    a = cross_val_score(model,feature,label,scoring=metrics.make_scorer(classification_report_with_accuracy_score),cv=10).mean()
    return round(a,3)
# all the models we used
RF = RandomForestClassifier()
KNN = KNeighborsClassifier()
LR = LogisticRegression(class_weight="balanced",solver="liblinear")
SV = SVC()
model_list = [RF,KNN,LR,SV]
for i in model_list:
    print(classification(i,feature,label))
    
# -------------------------------Bayesian Optimization---------------------------------------- #
# model optimization here
# -------random forest------- #
def rf_cv(n_estimators, min_samples_split, max_depth, max_features):
    val = cross_val_score(RandomForestClassifier(n_estimators=int(n_estimators),
                          min_samples_split=int(min_samples_split),
                          max_depth = int(max_depth),
                          max_features = min(max_features,0.999),
                          random_state = 2),
            feature,label,scoring=metrics.make_scorer(classification_report_with_accuracy_score),cv=10).mean()
    return val
rf_bo = BayesianOptimization(rf_cv,
                             {
                                 "n_estimators":(10,250),
                                 "min_samples_split":(2,25),
                                 "max_features":(0.1,0.999),
                                 "max_depth":(5,15)
                             })
num_iter = 15
init_points = 5
rf_bo.maximize(init_points=init_points,n_iter=num_iter)
# get the max params and max accuracy
print(rf_bo.max)

# -------KN neighbors-------- #
def knn_cv(n_neighbors):
    val = cross_val_score(KNeighborsClassifier(n_neighbors=int(n_neighbors)),
            feature,label,scoring=metrics.make_scorer(classification_report_with_accuracy_score),cv=10).mean()
    return val
knn_bo = BayesianOptimization(knn_cv,{
    "n_neighbors":(1,15)
    })
num_iter = 10
init_points = 5
knn_bo.maximize(init_points=init_points,n_iter=num_iter)
print(knn_bo.max)
# get the max params and max accuracy


# ------------------SVM---------------- #
def svc_cv(C, gamma):
    val = cross_val_score(SVC(C=10**C,
                               gamma=10**gamma,
                               random_state=2,
                               probability=True),feature,label,cv = 10,scoring=metrics.make_scorer(classification_report_with_accuracy_score)).mean()
    return val
svcBO = BayesianOptimization(svc_cv, {'C': (-5, 5),
                                     'gamma': (-5, 0)})
svcBO.maximize(init_points=10, n_iter=20)
print(svcBO.max)
# get the max params and max accuracy

# -------------Logistic Regression----------#
def LR_cv(C):
    val = cross_val_score(LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42, C=C),feature,label,cv = 10,scoring=metrics.make_scorer(classification_report_with_accuracy_score)).mean()
    return val
lr_bo = BayesianOptimization(LR_cv,{'C': (1, 1.5)},verbose=2,random_state=1)
lr_bo.maximize(n_iter=10)
print(lr_bo.max)
# get the max params and max accuracy






