# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:28:50 2016

@author: juju
"""

from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=3,max_leaf_nodes=5)
clf1=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,max_leaf_nodes=5)

clf=clf.fit(iris.data,iris.target)
clf1=clf1.fit(iris.data,iris.target)

tree.export_graphviz(clf,out_file="tree_Gini.dot")
tree.export_graphviz(clf1,out_file="tree_Entropie.dot")