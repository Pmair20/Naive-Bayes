#!/usr/bin/python

"""     
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
clf = GaussianNB()

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time()
predicted = clf.predict(features_test)
print ("Prediction time:", round(time()-t1, 3), "s")
print("Accuracy:",metrics.accuracy_score(labels_test, predicted))



