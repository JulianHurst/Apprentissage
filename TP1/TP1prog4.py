# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:19:20 2016

@author: juju
"""

from sklearn.datasets import load_iris
irisData = load_iris()

X=irisData.data
Y=irisData.target

from sklearn import neighbors

from sklearn.cross_validation import KFold

#Always put shuffle=True because of names
kf=KFold(len(X),n_folds=10,shuffle=True)
#kf=KFold(len(X),n_folds=3,shuffle=False)
scores=[]

for k in range(1,30):
    score=0
    clf = neighbors.KNeighborsClassifier(k)
    for learn,test in kf:
        print learn
        X_train=[X[i] for i in learn]
        Y_train=[Y[i] for i in learn]
        clf.fit(X_train,Y_train)
        X_test=[X[i] for i in test]
        Y_test=[Y[i] for i in test]
        score = score + clf.score(X_test,Y_test)
    scores.append(score)

print scores

print "meilleure valeur pour k : ",scores.index(max(scores))+1