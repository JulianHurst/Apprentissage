# -*- coding: utf-8 -*-

import random

from sklearn.datasets import load_iris
irisData=load_iris() # jeu de données Iris


X=irisData.data # les instances
Y=irisData.target # les classes

from sklearn import neighbors

from sklearn.cross_validation import KFold

# création d'une partition aléatoire de 10 folds sur l'intervalle [0, len(X)-1]
kf=KFold(len(X),n_folds=10,shuffle=True)

scores=[] # liste de scores


for k in range(1,30): # k est un paramètre du modèle : on cherche une valeur optimale entre 0 et 29
    score=0 # contiendra la somme des scores calculés sur chaque fold
    clf = neighbors.KNeighborsClassifier(k) # clf est un classifieur des k-ppv
    for learn,test in kf: # pour chaque fold
        X_train=[X[i] for i in learn]
        Y_train=[Y[i] for i in learn]
        clf.fit(X_train, Y_train) # entrainement sur l'échantillon d'apprentissage
        X_test=[X[i] for i in test]
        Y_test=[Y[i] for i in test]
        score = score + clf.score(X_test,Y_test) # évaluation de l'erreur sur l'échantillon test
    scores.append(score)

print(scores)

print("meilleure valeur pour k : ",scores.index(max(scores))+1) # meilleure valeur de $k$
