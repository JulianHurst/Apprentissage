# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:40:18 2016

@author: juju
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:00:49 2016

@author: juju
"""

import random

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
nb_voisins = 15

irisData=load_iris()

clf = neighbors.KNeighborsClassifier(nb_voisins)

X=irisData.data
Y=irisData.target

x=1
y=2

colors=["red","green","blue"]

"""
for i in range(3):
    pl.scatter(X[Y==i][:,x],X[Y==i][:,y],color=colors[i],\
    label=irisData.target_names[i])

pl.legend()
pl.xlabel(irisData.feature_names[x])
pl.ylabel(irisData.feature_names[y])
pl.title(u"Données Iris - dimension des sépales uniquement")
pl.show()
"""

X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.3,random_state=random.seed())

clf.fit(X_train,Y_train)
print "score test "+str(clf.score(X_test,Y_test))
