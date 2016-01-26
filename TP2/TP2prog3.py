# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:37:05 2016

@author: juju
"""

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import random
from sklearn import tree
X,Y=make_classification(n_samples=100000,n_features=20,n_informative=15,n_classes=3)
X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.3,random_state=random.seed())
for i in range(1,20):
    clf=tree.DecisionTreeClassifier(max_leaf_nodes=500*i)
    clf=clf.fit(X_train,Y_train)    
    print clf.score(X_train,Y_train), clf.score(X_test,Y_test)
#tree.export_graphviz(clf,out_file="essai.dot")