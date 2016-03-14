# -*- coding: utf-8 -*-
#@author: Julian Hurst

#2.3

import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

def regression(data):
    X=[]
    Y=[]
    V=[]
    for e in data:
        el=list(e[0])   #on transforme le tuple en liste
        el=[1]+el       #on ajoute le vecteur 1
        X.append(el)
        Y.append(e[1])
    XT=np.transpose(X)
    a=np.dot(XT,X)
    a=np.linalg.inv(a)
    w=np.dot(a,XT)
    w=np.dot(w,Y)
    print w
    return w

from sklearn.datasets import load_boston
boston = load_boston()
X=boston.data
Y=boston.target

clf=Ridge()
clf.fit(X,Y)
print "Coefficients Ridge : "
print clf.coef_

clfl=Lasso()
clfl.fit(X,Y)
print "Coefficients Lasso : "
print clfl.coef_

data=[]
for i in range(len(X)):
    data.append(((X[i]),Y[i]))

clfr=LinearRegression()
clfr.fit(X,Y)
print "Coefficients Régression linéaire par moindres carrés : "
print clfr.coef_

regression(data)

print "Erreur de prédiction pour la régression linéaire par moindres carrés : ",mean_squared_error(clfr.predict(X),Y)
print "Erreur de prédiction pour Ridge : ",mean_squared_error(clf.predict(X),Y)
print "Erreur de prédiction pour Lasso : ",mean_squared_error(clfl.predict(X),Y)

from sklearn.grid_search import GridSearchCV
alphas = np.logspace(-3, -1, 20)
for Model in [Ridge, Lasso]:
    gscv = GridSearchCV(Model(), dict(alpha=alphas), cv=5).fit(X, Y)
    print(Model.__name__, gscv.best_params_)

"""
4.
La régression Ridge a des coefficients de pondération plus élevés que ceux de la régression Lasso

5.
L'erreur de prédiction pour la régression linéaire est la moins élevée et celle pour Lasso est la plus élevée.
On peut en déduire que Ridge et Lasso évitent le sur-apprentissage et par conséquent ont une erreur de prédiction
plus élevées

6.
Les meilleurs paramètres renvoyés lors de la méthode de cross-validation sur une grille de valeurs pour les deux
modèles sont les mêmes : 0.10000000000000001
"""
