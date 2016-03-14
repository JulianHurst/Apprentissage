# -*- coding: utf-8 -*-
#@author: Julian Hurst

#2.1

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

dataR=np.loadtxt("dataRegLin2D.txt")
print dataR[0]
d=np.transpose(dataR)

l=[]

for i in range(len(d[0])):
    l.append([d[0][i],d[1][i]])

print l

regr=linear_model.LinearRegression()

regr.fit(l,d[2])

print "Y=",d[2]
print('Variance score: %.2f' % regr.score(l, d[2]))

print regr.predict(l)

print mean_squared_error(regr.predict(l),d[2])
