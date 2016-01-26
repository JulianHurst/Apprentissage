# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:16:00 2016

@author: juju
"""

from sklearn.datasets import load_digits
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
import pylab as pl
import numpy as np

from sklearn.cross_validation import KFold
import random

digits = load_digits()
X=digits.data
Y=digits.target

scores=[]
kf=KFold(len(X),n_folds=8,shuffle=True)

for k in range(1,30):
    score=0
    clf = neighbors.KNeighborsClassifier(k)
    for learn,test in kf:
        X_train=[X[i] for i in learn]
        Y_train=[Y[i] for i in learn]

        try:
            clf.fit(X_train[:100],Y_train[:100])
        except ValueError as e:
            print "Error! "+str(e)
            print np.array(X_train).shape
            print np.array(Y_train).shape
            exit(1)
          
        X_test=[X[i] for i in test]
        Y_test=[Y[i] for i in test]
        score = score + clf.score(X_test,Y_test)        
    scores.append(score)

print "meilleure valeur pour k : ",scores.index(max(scores))+1

clf = neighbors.KNeighborsClassifier(scores.index(max(scores))+1)

X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.3,random_state=random.seed())

clf.fit(X_train,Y_train)
print 1-clf.score(X_test,Y_test)

num=0
pl.gray()
for i in range(0,len(Y_test)-1):    
        if Y_train[i]!=Y_test[i] and num<4:
            print str(Y_train[i])+" devrait être "+str(Y_test[i])
            pl.matshow(digits.images[Y_train[i]])
            num+=1

pl.show()
