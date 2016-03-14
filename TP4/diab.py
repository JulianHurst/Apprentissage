# -*- coding: utf-8 -*-
#@author: Julian Hurst

#2.2

import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()


X_d=diabetes.data
Y_d=diabetes.target

regr_d=linear_model.LinearRegression()

regr_d.fit(X_d,Y_d)

print regr_d.coef_

boston = datasets.load_boston()

X_b=boston.data
Y_b=boston.target

regr_b=linear_model.LinearRegression()

regr_b.fit(X_b,Y_b)

print regr_b.coef_
