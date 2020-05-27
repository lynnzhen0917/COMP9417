import numpy as np
import csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn import model_selection
import pandas as pd
path1 = "F://9417/feature_label_2/panas_P.csv"
path2 = "F://9417/feature_label_2/panas_N.csv"
path3 = "F://9417/feature_label_2/flourishing.csv"  

data=pd.read_csv(path1)
# seperate the file to feature and label dataframes
feature = data.iloc[:,1:17]
label = data.iloc[:,17:18]
# ---------------------------GET DATA---------------------#
# Get data from data achieved 
filename = 'pre_result(1)(1).csv'
raw_data = []
with open(filename) as f:
    for line in csv.reader(f):
        raw_data.append(line)  
        
# pre-process featuer 
data_str_x = np.array(raw_data[1:])[:,1:-6]  

data_str_y = np.array(raw_data[1:])[:,-6:]  # make sure last 6 columns are y values
num_x = data_str_x.shape[1]             
x_all = data_str_x.astype(float)    
scaler = MinMaxScaler()
scaler.fit(x_all)
x_processed = scaler.transform(x_all)
y_classified = data_str_y.astype(float) 

#print scatter plot of each feature 
def do_scatter(x_processed):
    X = np.arange(1, x_processed.shape[0]+1)
    for i in range(x_processed.shape[1]):
        Y = x_processed[:,i]
        plt.xlabel("uid")
        plt.ylabel("Value")
        plt.title("Scatter plot for feature " + raw_data[0][i+1])    
        plt.scatter(X, Y, s=75,alpha=.5)
        plt.show()    
        print("Avarage：%f" % np.mean(Y))    
        print("Median：%f" % np.median(Y))
        print("Std：%f" % np.std(Y))
    return 

#--------------------------------SVC feature select--------------------------- #
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

def svc_feature_select(feature,label):
    # feature selection svc
    X, y = feature,label
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=10,
                  scoring='accuracy')
    rfecv.fit(X, y)
    X_new = rfecv.transform(X)

    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    print(rfecv.grid_scores_)
    print(rfecv.support_)
    print(rfecv.ranking_)
    return None 
# --------------------------Logistic Regression--------------------------------#
from sklearn.linear_model import LogisticRegression
def LogisticRegrssion_feature_select(feature,label):
    
    # feature selection LG
    X, y = feature,label
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10,
                  scoring='accuracy')
    rfecv.fit(X, y)
    X_new = rfecv.transform(X)

    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    print(rfecv.grid_scores_)
    print(rfecv.support_)
    print(rfecv.ranking_)
    return None
# -----------------------------Tree Feature Select-----------------------------#
def tree_feature_select(feature,label):
    # feature selection RF
    X, y = feature,label
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    print(clf.feature_importances_.max())
    return None

# -------------------------------KNN feature select------------------------------#
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def KNN_feature_selection(feature,label):
    knn = KNeighborsClassifier(n_neighbors=5)
    sfs1 = SFS(estimator=knn, 
               k_features=5,
               forward=True, 
               floating=False, 
               scoring='accuracy',
               cv=5)

    pipe = Pipeline([('sfs', sfs1), 
                     ('knn', knn)])

    param_grid = [
      {'sfs__k_features': [1,2,3,4],
       'sfs__estimator__n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
      ]

    gs = GridSearchCV(estimator=pipe, 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      n_jobs=1, 
                      cv=5,
                      iid=True,
                      refit=False)

    # run gridearch
    gs = gs.fit(feature, label)

  
    print('Best features:', gs.best_estimator_.steps[0][1].k_feature_idx_)

    for i in range(len(gs.cv_results_['params'])):
        print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])

    print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
    #plot_sfs(sfs1.get_metric_dict(), kind='std_err')

    # using PCA
    from sklearn.decomposition import PCA
    pca=PCA(n_components='mle',copy=True)
    newX=pca.fit_transform(X)
    print(newX)



    from sklearn.feature_selection import SelectPercentile, chi2
    X, y = feature,label
    
    selector = SelectPercentile(chi2, percentile=15)
    X_new = selector.fit_transform(X, y)
    print('Filtered data shape:', X_new.shape)
    print(selector.get_support())
    print('F-Scores:', selector.scores_)
    print(selector.scores_.max())



    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    estimators = [('reduce_dim', PCA()), ('KNN', KNeighborsClassifier())]
    pipe = Pipeline(estimators)
    pipe.fit(X,y)
    prediction = pipe.predict(X)
    #print(pipe["reduce_dim"].components_)
    return None 

tree_feature_select(feature,label)
LogisticRegrssion_feature_select(feature,label)
svc_feature_select(feature,label)
KNN_feature_select(feature,label)


# -------------fucntion to print together----------#
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#这里导入你自己的数据
#......
#......
#x_axix，train_pn_dis这些都是长度相同的list()

#开始画图

path1 = "/Users/lilily/Desktop/9417/Assignments/Project/Panas_N.csv"
path2 = "/Users/lilily/Desktop/9417/Assignments/Project/Panas_P.csv"
path3 = "/Users/lilily/Desktop/9417/Assignments/Project/flourishing.csv"
def get_input(path):
    data=pd.read_csv(path)
# seperate the file to feature and label dataframes
    feature = data.iloc[:,1:-1]
    lable = data.iloc[:,-1]
    return feature, lable

feature, lable = get_input(path1)
#print(lable)
x, Y = svc_feature_select(feature,lable)
feature, lable = get_input(path2)
x1, Y1 = svc_feature_select(feature,lable) 
feature, lable = get_input(path3)
x2, Y2 = svc_feature_select(feature,lable)
# sub_axix = filter(lambda x:x%200 == 0, x)

plt.plot(x, Y, color='red', label='Panas_N')
plt.plot(x1, Y1,  color='blue', label='Panas_P')
plt.plot(x2, Y2,   color='green', label='flourishing')
# plt.plot(x_axix, thresholds, color='blue', label='threshold')
plt.legend() # 显示图例

plt.xlabel('Number of feature selected')
plt.ylabel('Cross validation score')
plt.show()





