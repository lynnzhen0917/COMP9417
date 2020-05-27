import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score,precision_score

data=pd.read_csv("F:\9417\result.csv",header=None)

#pre-processing
def normalisation(xlist):
     xmin = min(xlist)
     xmax = max(xlist)
     for i in range(len(xlist)):
         xlist[i] = (xlist[i] - xmin)/(xmax - xmin)
         #print(xlist)
     return xlist

list1 = []
for i in range(len(x)):
    list1.append(normalisation(x[i]))
list2 = [[row[i] for row in list1] for i in range(690)]

#creating test and training sets
x_training = np.array(list2[0:621])
y_training = np.array(y[0:621])
x_test = np.array(list2[621:690])
y_test = np.array(y[621:690])

#Part A get two accuracy
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_training,y_training)
print("Accuracy for training set: ",knn.score(x_training,y_training))
print("Accuracy2 for test set: ",knn.score(x_test,y_test))
#Accuracy:  0.8969404186795491
#Accuracy2:  0.7681159420289855

#Part B AUC score for training and test sets
#Part C plot them
neighbors = np.arange(1,31)

def auclist(rangelist,xlist,ylist,xtest,ytest):
    auclist_train = np.empty(len(rangelist))
    auclist_test = np.empty(len(rangelist))
    for i,k in enumerate(rangelist):
        knn_n = KNeighborsClassifier(n_neighbors=k)
        knn_n.fit(xlist,ylist)
        y_pred = knn_n.predict_proba(xlist)
        y_pred2 = knn_n.predict_proba(xtest)
        auclist_train[i] = roc_auc_score(ylist,y_pred[:,1])
        auclist_test[i] = roc_auc_score(ytest,y_pred2[:,1])
    return auclist_train,auclist_test
auclist_train,auclist_test = auclist(neighbors,x_training,y_training,x_test,y_test)
auclist_testlist = auclist_test.tolist()

print("the optimal value is: ",auclist_testlist.index(max(auclist_testlist))+1)


fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(range(1,31), auclist_train, "p:",label="training", color='r')
ax2.plot(range(1,31), auclist_test, "p:",label="test", color='k')
plt.show()



#Part D precision and recall for k=5 and k=2
def precision_and_recall(k,xlist,ylist,xtest,ytest):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xlist,ylist)
    y_pred = knn.predict(xtest)
    return precision_score(ytest,y_pred),recall_score(ytest,y_pred)

print("precision and recall for k = 5 is: ",precision_and_recall(5,x_training,y_training,x_test,y_test))
print("precision and recall for k = 2 is: ",precision_and_recall(2,x_training,y_training,x_test,y_test))
