# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 16:10:16 2016

@author: Julian Hurst
"""

import numpy as np
import pylab as pl
import random
from sklearn.datasets import load_iris

#Perceptron binaire
def perceptron(data):    
    ps=0        
    w=[] 
    we=0
    sv=False    
    for _ in range(len(data[0][0])):
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
            except Exception:   # si la taille des données est de 1
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

#1. Perceptron multi-classe
def multiperceptron(data,m):    
    ps=0        
    w=[] 
    wl=[]
    argm=[]
    we=0
    sv=False
    for _ in range(m):        
        for _ in range(len(data[0][0])):
            w.append(0)
        wl.append(w)
        w=[]    
    for _ in range(1000):
        for e in data:            
            for k in range(m):  # pour toute classe k=1..m
                for j in range(len(e[0])):
                    ps+=wl[k][j]*e[0][j]
                argm.append(ps)
            amax = np.argmax(argm) 
                          
            if e[1]!=amax:                    
                for j in range(len(e[0])):                  
                        wl[e[1]][j]+=e[0][j]
                        wl[amax][j]-=e[0][j]
            ps=0
            argm=[]
    if sv:
        print "we : ",we
        return we
    else:
        print "wl : ",wl
        return wl

#Géneration de données pour 2 classes
def genererDonnees(n):
    xb=(pl.rand(n)*2-1)/2-0.5    
    yb=(pl.rand(n)*2-1)/2+0.5
    xr=(pl.rand(n)*2-1)/2+0.5
    yr=(pl.rand(n)*2-1)/2-0.5    
    donnees=[]
    for i in range(len(xb)):
        donnees.append(((xb[i],yb[i]),False))
        donnees.append(((xr[i],yr[i]),True))
    return donnees

#2. Géneration de données pour plus de 2 classes
def multigenererDonnees(n,m):
    xb=(pl.rand(n)*2-1)/2-0.5    
    yb=(pl.rand(n)*2-1)/2+0.5
    xr=(pl.rand(n)*2-1)/2+0.5
    yr=(pl.rand(n)*2-1)/2-0.5    
    donnees=[]
    for i in range(len(xb)):
        k=random.randint(0,m-1)
        donnees.append(((xb[i],yb[i]),k))
        k=random.randint(0,m-1)
        donnees.append(((xr[i],yr[i]),k))
    return donnees
    
#2. Jeu de données générée
print "Données générées:"
data_train=multigenererDonnees(100,3)
data_test=multigenererDonnees(100,3)
multiperceptron(data_train,3)


#3. Jeu de données Iris
print "Iris:"
irisData=load_iris()
X=irisData.data
Y=irisData.target
data=[]
for i in range(len(X)):
    data.append(((X[i]),Y[i]))

multiperceptron(data,3)    



        
