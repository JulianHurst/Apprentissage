# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:35:18 2016

@author: juju
"""

import perceptron_data as p
import numpy as np
import pylab as pl
import types

def perceptron(data):    
    ps=0        
    w=[] 
    we=0
    sv=False
    for _ in range(len(data[0])):
        w.append(0)
    for _ in range(1000):
        for e in data:
            try:
                for j in range(len(e[0])):
                    ps+=w[j]*e[0][j]            
                if ps<0:
                    yc=False
                else:
                    yc=True            
                if e[1]!=yc:
                    for j in range(len(e[0])):
                        if e[1]:
                            w[j]+=e[0][j]
                        else:
                            w[j]-=e[0][j]
                ps=0    
            except Exception:
                sv=True
                ps+=we*e[0]
                if ps<0:
                    yc=False
                else:
                    yc=True            
                if e[1]!=yc:                    
                    if e[1]:
                        we+=e[0]
                    else:
                        we-=e[0]  
    if sv:
        print "we : ",we
        return we
    else:
        print "w : ",w
        return w

def genererDonnees(n):
    xb=(pl.rand(n)*2-1)/2-0.5    
    yb=(pl.rand(n)*2-1)/2+0.5
    xr=(pl.rand(n)*2-1)/2+0.5
    yr=(pl.rand(n)*2-1)/2-0.5
    print xb
    donnees=[]
    for i in range(len(xb)):
        donnees.append(((xb[i],yb[i]),False))
        donnees.append(((xr[i],yr[i]),True))
    return donnees
    

data_train=genererDonnees(100)
data_test=genererDonnees(100)

perceptron(data_train)
#perceptron(p.bias)

#for i in range(len(data_train[0])):
 ##   pl.scatter(data_train[0][i],data_train[0][i],color="red")
   # pl.scatter(data_test[0][i],data_test[0][i],color="blue")
for i in data_test:
    if i[1]:
        pl.scatter(i[0][0],i[0][1],color="blue")
    else:
        pl.scatter(i[0][0],i[0][1],color="red")
pl.show()


        