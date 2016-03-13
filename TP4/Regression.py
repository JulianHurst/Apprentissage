# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:35:05 2016

@author: Julian Hurst
"""

import numpy as np
from sklearn.datasets import load_iris

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

irisData=load_iris()
X=irisData.data
Y=irisData.target


data=[]
for i in range(len(X)):
    data.append(((X[i]),Y[i]))

regression(data)

dataR=np.loadtxt("dataRegLin2D.txt")
print "hello ",dataR[0]
d=np.transpose(dataR)
print "hi ",d[0]

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
fig = plt.figure()
ax = Axes3D(fig)
for i in range(len(d[0])):
    ax.scatter(d[0][i],d[1][i],d[2][i])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
plt.show()

for i in range(len(d[0])):
    pl.scatter(d[0][i],d[2][i])

pl.show()

for i in range(len(d[0])):
    pl.scatter(d[1][i],d[2][i])

pl.show()
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl

data2=[]
for i in range(len(d[0])):
    data2.append(((d[0][i],d[1][i]),d[2][i]))

w=[]
w=regression(data2)

for i in range(len(d[0])):
    pl.scatter(d[1][i],d[2][i])
    #plt.quiver(w[0],w[1],w[2])
pl.show()

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#scatter(dataR[0],dataR[1],)
