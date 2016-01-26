# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:43:18 2016

@author: juju
"""

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
X,Y=make_classification(n_samples=100000,n_informative=15,n_features=20,n_classes=3)
X_1,X_2,Y_1,Y_2=\
train_test_split(X,Y,test_size=0.95,random_state=random.seed())

X_app,X_test,Y_app,Y_test=train_test_split(X_1,Y_1,test_size=0.8,random_state=random.seed())


