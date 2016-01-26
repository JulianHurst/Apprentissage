# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:29:58 2016

@author: juju
"""
"""
n_samples nb examples
n_features nb d'attributs  X=R^n    n=nb d'attributs par ex : R^20 -> 20 attr
n_informative nb de paramètres(features) "utiles"
n_redundant nb de features redondants(plus ou moins identiques)
n_repeated nb de features identiques
n_classes nb de classes ou labels du pb
n_clusters_per_class nb de groupes de classes 
"""

from sklearn.datasets import make_classification
X,Y=make_classification(n_samples=200,n_features=2,n_redundant=0,\
n_clusters_per_class=1,n_classes=3)
import pylab as pl
pl.scatter(X[:,0],X[:,1],c=Y)
pl.show()