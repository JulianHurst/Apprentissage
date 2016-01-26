# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:44:42 2016

@author: juju
"""

import sys

from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree

if len(sys.argv)==1:
    clf=tree.DecisionTreeClassifier()
else:
    clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,max_leaf_nodes=int(sys.argv[1]))
    
"""
for i in range(3,9):
    clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,max_leaf_nodes=i)
    tree.export_graphviz(clf,out_file="essai"+str(i)+".dot")
"""

clf=clf.fit(iris.data,iris.target)
print clf.predict(iris.data[50,:])
print clf.score(iris.data,iris.target)
tree.export_graphviz(clf,out_file="essai.dot")