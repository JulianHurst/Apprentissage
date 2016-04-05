# -*- coding: utf-8 -*-
"""Ce fichier contient la correction du TP1 

.. moduleauthor:: François Denis
"""

# ouverture du fichier de données digits 
from sklearn.datasets import load_digits
digits=load_digits()


X = digits.data
Y = digits.target

from sklearn.cross_validation import train_test_split
import random

# répartition des données en un ensemble d'apprentissage et un ensemble test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

# préparation de la validation croisée ...
from sklearn.cross_validation import KFold
kf=KFold(len(X_train),n_folds=10,shuffle=True)

scores=[]

# pour sélectionner le paramètre optimal k
from sklearn import neighbors
for k in range(1,30):
    score=0
    clf = neighbors.KNeighborsClassifier(k)
    for learn,test in kf:
        X_train_val=[X_train[i] for i in learn]
        Y_train_val=[Y_train[i] for i in learn]
        clf.fit(X_train_val, Y_train_val)
        X_test_val=[X_train[i] for i in test]
        Y_test_val=[Y_train[i] for i in test]
        score = score + clf.score(X_test_val,Y_test_val)
    scores.append(score)

# valeur optimale de k : 
k_opt=scores.index(max(scores))+1


# affichage de tous les scores. On constate :
# - que les scores correspondant aux petites valeurs de k (<= 5 ou  10) sont proches
# - que les scores diminuent sensiblement pour des valeurs de k supérieures

for s in scores:
    print("%5.3f " %s, end='')

# on trouve généralement des valeurs de k assez petites (<= 5 ou 10)
print("valeur optimale de  k : ",k_opt)

clf = neighbors.KNeighborsClassifier(k_opt)

clf.fit(X_train, Y_train)

# on constate que le score est très proche de 1
print("score sur l'échantillon test : ","%5.3f " %clf.score(X_test,Y_test))

import pylab as pl

pl.gray()

print("Quelques images mal classées")

for i in range(len(X)):
    [c]=clf.predict(X[i])
    if c!=Y[i]:
        pl.matshow(digits.images[i])
        print("classe réelle : ",Y[i],"classe prédite : ",c)
        pl.show()

# on constate que les images mal classées sont effectivement difficiles à prédire et que la classe prédite par le classifieur est souvent plausible
