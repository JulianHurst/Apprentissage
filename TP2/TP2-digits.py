# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:28:22 2016

@author: juju
"""

from sklearn.datasets import load_digits

from sklearn import tree

import math
from sklearn.metrics import confusion_matrix


digits = load_digits()
X=digits.data
Y=digits.target

clf=tree.DecisionTreeClassifier() #Gini

from sklearn.cross_validation import KFold

# création d'une partition aléatoire de 10 folds sur l'intervalle [0, len(X)-1]
kf=KFold(len(X),n_folds=10,shuffle=True)

scores=[] # liste de scores


for k in range(1,30): # k est un paramètre du modèle : on cherche une valeur optimale entre 0 et 29
    score=0 # contiendra la somme des scores calculés sur chaque fold
    #clf = neighbors.KNeighborsClassifier(k) # clf est un classifieur des k-ppv
    for learn,test in kf: # pour chaque fold
        X_train=[X[i] for i in learn]
        Y_train=[Y[i] for i in learn]
        clf.fit(X_train, Y_train) # entrainement sur l'échantillon d'apprentissage
        X_test=[X[i] for i in test]
        Y_test=[Y[i] for i in test]
        score = score + clf.score(X_test,Y_test) # évaluation de l'erreur sur l'échantillon test
    scores.append(score)

print(scores)

print "meilleure valeur pour k (Gini) : ",scores.index(max(scores))+1 # meilleure valeur de $k$

clf.fit(X_train, Y_train)
e=clf.score(X_test,Y_test)
N=len(X_test)
s=1.96*math.sqrt(e*(1-e)/N)
f=clf.score(X,Y)
if f<e-s or f> e+s:    
    print "Gini",f, e-s,e+s

Y_pred =clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print cm

n00=0
n11=0
same=0
for i in cm:
    for j in range(len(i)):
        if j==same:
            n11+=i[j]
        else:
            n00+=i[j]
    same+=1

print "Nombre d'exemples que Gini classe correctement : ",n00
print "Nombre d'exemples que Gini classe incorrectement : ",n11

clf=tree.DecisionTreeClassifier(criterion='entropy') #Entropy

scores=[] # liste de scores


for k in range(1,30): 
    score=0 # contiendra la somme des scores calculés sur chaque fold
    for learn,test in kf: # pour chaque fold
        X_train=[X[i] for i in learn]
        Y_train=[Y[i] for i in learn]
        clf.fit(X_train, Y_train) # entrainement sur l'échantillon d'apprentissage
        X_test=[X[i] for i in test]
        Y_test=[Y[i] for i in test]
        score = score + clf.score(X_test,Y_test) # évaluation de l'erreur sur l'échantillon test
    scores.append(score)

print(scores)

print "meilleure valeur pour k (Entropy) : ",scores.index(max(scores))+1 # meilleure valeur de $k$

clf.fit(X_train, Y_train)
e=clf.score(X_test,Y_test)
N=len(X_test)
s=1.96*math.sqrt(e*(1-e)/N)
f=clf.score(X,Y)
if f<e-s or f> e+s:    
    print "Entropy",f, e-s,e+s
    
Y_pred =clf.predict(X_test)
cme = confusion_matrix(Y_test, Y_pred)
print cme

ne00=0
ne11=0
same=0
for i in cme:
    for j in range(len(i)):
        if j==same:
            ne11+=i[j]
        else:
            ne00+=i[j]
    same+=1

print "Nombre d'exemples que Entropy classe correctement : ",ne00
print "Nombre d'exemples que Entropy classe incorrectement : ",ne11

print "n00 : ",n00+ne00
print "n11 : ",n11+ne11

same=0
n10=0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if cme[i][j]>cm[i][j] and j!=same:
            n10=n10+(cme[i][j]-cm[i][j])            
    same+=1

print "Nombre d'exemples que le premier classifieur classe correctement et le second incorrectement : ",n10

same=0
n01=0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if cme[i][j]<cm[i][j] and j!=same:
            n01=n01+(cm[i][j]-cme[i][j])            
    same+=1

print "Nombre d'exemples que le second classifieur classe correctement et le premier incorrectement : ",n01

comp=((abs(n01-n10)-1)**2)/float((n01+n10))
print "Le test de McNemar nous fournit le résultat : ",comp
    

#Intervalle de confiance
#On peut déduire des intervalles de confiance et de l'erreur estimée des classifieurs qu'on ne peut être sûrs à 95% que l'erreur estimée est bonne car elle n'est pas dans l'intervalle de confiance.

#Test de McNemar
#Il est impossible de dire si l'un des deux classifieurs est meilleur que l'autre car le nombre calculé est toujours inférieur à 3.841459