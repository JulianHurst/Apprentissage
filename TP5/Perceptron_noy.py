# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 06:24:19 2016

@author: Julian Hurst
"""

import numpy as np
from numpy.linalg import norm
from sklearn.datasets import load_iris
import pylab as pl

def noyauGaussien(x1,x2,sigma):
        Y=np.subtract(x1,x2)
        N=norm(Y)**2
        E=-N/(sigma**2)
        return np.exp(E)
    
def noyauPolynomial(x1,x2,k):
    return (1+np.inner(x1,x2))**k    
    
#3.

def learnKernelPerceptron(data,target,kernel,h):    
    a=np.zeros(len(data))
    if kernel==1:
        kp=noyauGaussien
    else:
        kp=noyauPolynomial    
    for p in range(10):
        print p 
        for i in range(len(data)):            
            som=0
            for j in range(len(data)):
                som+=a[j]*target[j]*kp(data[j],data[i],h)            
            y=np.sign(som)            
            if y!=target[i]:
                a[i]+=1            
    return a

def predictKernelPerceptron(kp,x,data,kernel,h):
    

def genererDonnees(n):
    xb=(pl.rand(n)*2-1)/2-0.5    
    yb=(pl.rand(n)*2-1)/2+0.5
    xr=(pl.rand(n)*2-1)/2+0.5
    yr=(pl.rand(n)*2-1)/2-0.5    
    donnees=[]
    for i in range(len(xb)):
        donnees.append(((xb[i],yb[i]),-1))
        donnees.append(((xr[i],yr[i]),1))
    return donnees
    
data=genererDonnees(100)
X=[]
Y=[]
for x in data:
   X.append(x[0])
   Y.append(x[1])
   
print learnKernelPerceptron(X,Y,0,10)
    
print noyauGaussien([1,5,4],[0,3,2],2)

print noyauPolynomial([1,2,3],[2,3,4],2) 
