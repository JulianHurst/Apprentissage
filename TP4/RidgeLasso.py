# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression

def regression(data):
    X=[]
    Y=[]
    V=[]
    """
    for _ in range(len(data[0][0])):
        V.append(1)
    """
    #print V
    #X.append(V)
    for e in data:
        el=list(e[0])   #on transforme le tuple en liste
        el=[1]+el       #on ajoute le vecteur 1
        X.append(el)
        Y.append(e[1])
    #print X
    XT=np.transpose(X)
    #print XT
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

print X
print Y

clf=Ridge()
clf.fit(X,Y)
print "Coefficients Ridge : "
print clf.coef_

clfl=Lasso()
clfl.fit(X,Y)
print "Coefficients Lasso : "
print clfl.coef_

print "Score Ridge : ",clf.score(X,Y)
print "Score Lasso : ",clfl.score(X,Y)

data=[]
for i in range(len(X)):
    data.append(((X[i]),Y[i]))

clfr=LinearRegression()
clfr.fit(X,Y)
print clfr.coef_
print clfr.score(X,Y)

regression(data)

from sklearn.grid_search import GridSearchCV
alphas = np.logspace(-3, -1, 20)
for Model in [Ridge, Lasso]:
gscv = GridSearchCV(Model(), dict(alpha=alphas), cv=5).fit(X, y)
print(Model.__name__, gscv.best_params_)

"""
1.
La régression Ridge a des coefficients de pondération plus élevés que ceux de la régression Lasso

2.
Le score de la régression linéaire par moindres carrés est plus élevé que celle de Ridge qui est lui-même plus
élevé que celle de Lasso. On peut en déduire que pour ce jeu de données qu'il y a moins d'erreur
de prédiction pour la régression linéaire par moindres carrés que pour la régression Ridge ou Lasso

"""
