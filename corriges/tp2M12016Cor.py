# -*- coding: utf-8 -*-


import random
import math

# 1 point
from sklearn.datasets import load_digits
digits=load_digits()
X=digits.data
Y=digits.target

# 1 point
from sklearn.cross_validation import train_test_split
X_app,X_test,Y_app,Y_test=train_test_split(X,Y,test_size=0.30,random_state=random.seed())



# 1 point
from sklearn.cross_validation import KFold
kf=KFold(len(X_app),n_folds=10,shuffle=True)


from sklearn import tree

# 3 points
scoresGini = []
scoresEntropie = []

for k in range(1,20):
    scoreGini = 0
    scoreEntropie = 0
    clfGini=tree.DecisionTreeClassifier(max_leaf_nodes=500*k)
    clfEntropie=tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=500*k)
    for learn,test in kf:
        X_tr=[X_app[i] for i in learn]
        Y_tr=[Y_app[i] for i in learn]
        clfGini.fit(X_tr, Y_tr)
        clfEntropie.fit(X_tr, Y_tr)
        X_tt=[X_app[i] for i in test]
        Y_tt=[Y_app[i] for i in test]
        scoreGini = scoreGini + clfGini.score(X_tt,Y_tt)
        scoreEntropie = scoreEntropie + clfEntropie.score(X_tt,Y_tt)
    scoresGini.append(scoreGini)
    scoresEntropie.append(scoreEntropie)

# 1 point
max_leaf_nodesGini = 500*(scoresGini.index(max(scoresGini))+1)
max_leaf_nodesEntropie = 500*(scoresEntropie.index(max(scoresEntropie))+1)

# 1 points
print("Nombre total de feuilles sélectionné pour le critère de Gini : ", max_leaf_nodesGini)
print("Nombre total de feuilles sélectionné pour le critère Entropie : ",max_leaf_nodesEntropie)


# 2 points
clfGini=tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodesGini)
clfEntropie=tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=max_leaf_nodesEntropie)
clfGini.fit(X_app, Y_app)
clfEntropie.fit(X_app, Y_app)
eGini=clfGini.score(X_test,Y_test)
eEntropie=clfEntropie.score(X_test,Y_test)

# 2 point
print("Erreur estimée pour le critère de Gini :  %6.4f" %(1-eGini))
print("Erreur estimée pour le critère Entropie : %6.4f" %(1-eEntropie))

N=len(X_test)
sGini=1.96*math.sqrt(eGini*(1-eGini)/N)
sEntropie=1.96*math.sqrt(eEntropie*(1-eEntropie)/N)

# 3 points
print("Intervalle de confiance pour le critère de Gini :  [%6.4f" %(1-eGini-sGini),", %6.4f" %(1-eGini+sGini),"]")
print("Intervalle de confiance pour le critère Entropie :  [%6.4f" %(1-eEntropie-sEntropie),", %6.4f" %(1-eEntropie+sEntropie),"]")


# 2 points
from sklearn.metrics import confusion_matrix
Y_pred_Gini=clfGini.predict(X_test)
Y_pred_Entropie=clfEntropie.predict(X_test)
cmGini = confusion_matrix(Y_test, Y_pred_Gini)
cmEntropie = confusion_matrix(Y_test, Y_pred_Entropie)
print("Matrice de confusion pour l'arbre de décision appris avec le critère de Gini\n",cmGini)
print("Matrice de confusion pour l'arbre de décision appris avec le critère Entropie\n",cmEntropie)

# 3 points
n00 = n01 = n10 = n11 = 0
for i in range(len(X_test)):
    if Y_pred_Gini[i] == Y_test[i]:
        if Y_pred_Entropie[i] == Y_test[i]:
            n11 += 1
        else:
            n10 += 1
    elif Y_pred_Entropie[i] == Y_test[i]:
            n01 += 1
    else:
            n00 += 1

x = (abs(n01-n10)-1)**2/(n01+n10)

print("Critère de McNemar ", x)

if (x > 3.841459):
        if n10 > n01:
                print("le classifieur appris avec le critère de Gini est significativement meilleur que le classifieur appris avec le critère Entropie")
        else:
                print("le classifieur appris avec le critère Entropie est significativement meilleur que le classifieur appris avec le critère de Gini")
else:
        print("le test ne permet pas de départager les 2 classifieurs")



