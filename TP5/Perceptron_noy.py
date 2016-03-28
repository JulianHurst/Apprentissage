# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 06:24:19 2016

@author: Julian Hurst
"""

import numpy as np
from numpy.linalg import norm
from sklearn.datasets import load_iris
import pylab as pl
import tp5utils as utils
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import os

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
        #print p
        for i in range(len(data)):
            som=0
            for j in range(len(data)):
                som+=a[j]*target[j]*kp(data[j],data[i],h)
            y=np.sign(som)
            if y!=target[i]:
                a[i]+=1
    return a

def predictKernelPerceptron(kp,x,data,kernel,h):
    y=np.zeros(len(data))
    if kernel==1:
        noy=noyauGaussien
    else:
        noy=noyauPolynomial
    for p in range(10):
        #print p
        for i in range(len(data)):
            som=0
            for j in range(len(data)):
                som+=kp[j]*x[j]*noy(data[j],data[i],h)
            y[i]=np.sign(som)
    return y

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

def erreurapp(P,X,Y):
    ea = 0
    for i in range(len(X)):
        if (P[i] <> Y[i]):
            ea = ea+1
    return  1.*ea/len(X)

data=genererDonnees(100)
X=[]
Y=[]

for x in data:
   X.append(x[0])
   Y.append(x[1])

w=learnKernelPerceptron(X,Y,0,10)
print w
print predictKernelPerceptron(w,Y,X,1,10)
print Y


"""
#DONNEES VECTEURS IMAGES
V_P=utils.chargementVecteursImages('Data/Ailleurs','Data/Mer',1,-1,10)

Data_P=[]
Target_P=[]

for x in V_P[1]:
   Data_P.append(x.tolist()[0])
for x in V_P[2]:
   Target_P.append(x)
pixel=True
"""

#DONNEES VECTEURS HISTOGRAMMES
V_P=utils.chargementHistogrammesImages('Data/Ailleurs','Data/Mer',1,-1)

Data_P=[]
Target_P=[]

for x in V_P[1]:
   Data_P.append(x.tolist())
for x in V_P[2]:
   Target_P.append(x)
pixel=False

X_train,X_test,Y_train,Y_test=\
train_test_split(Data_P,V_P[2],test_size=0.5,random_state=random.seed())

#KP
print "KP"
u=learnKernelPerceptron(X_train,Y_train,0,0.1)

pred=predictKernelPerceptron(u,Y_test,X_test,0,0.1)
print "Erreur d'apprentissage : ",erreurapp(pred,X_test,Y_test)

print "Erreur réelle : ",mean_squared_error(predictKernelPerceptron(u,Y_test,X_train,0,0.1),Y_test)

print "\n"
print "P"
clfp=Perceptron(alpha=0.5)
clfp.fit(X_train,Y_train)
print "Erreur d'apprentissage : ",clfp.score(X_test,Y_test)
scores = cross_validation.cross_val_score(clfp,X_test,Y_test)
print scores
print "Erreur réelle : ",scores.mean()

print "\n"
print "SVC"
clfs=svm.LinearSVC(C=15)
clfs.fit(X_train,Y_train)
print "Erreur d'apprentissage : ",clfs.score(X_test,Y_test)
pred=clfs.predict(X_test)
scores = cross_validation.cross_val_score(clfs,X_test,Y_test)
print scores
print "Erreur réelle : ",scores.mean()

print "\n"
print "NEIGHBORS"
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train,Y_train)
print "Erreur d'apprentissage : ",neigh.score(X_test,Y_test)
scores = cross_validation.cross_val_score(neigh,X_test,Y_test)
print scores
print "Erreur réelle : ",scores.mean()

print "\n"
print "AD"
ad = DecisionTreeClassifier(max_leaf_nodes=20)
ad.fit(X_train,Y_train)
print "Erreur d'apprentissage : ",ad.score(X_test,Y_test)
scores = cross_validation.cross_val_score(ad,X_test,Y_test)
print scores
print "Erreur réelle : ",scores.mean()

print "\n"
L=[]
Pix=utils.importVecteursTest("BinTest/tPixel-60.npy")
for x in Pix:
    if pixel:
        L.append(x[:300])
    else:
        L.append(x[:768])
print "Prediction V Pixel : ",clfs.predict(L)



Hist=utils.importVecteursTest("BinTest/tHisto.npy")

del L[:]

for x in Hist:
    if pixel:
        L.append(x[:300])
    else:
        L.append(x[:768])

print "Prediction V Histogramme : ",clfs.predict(L)
