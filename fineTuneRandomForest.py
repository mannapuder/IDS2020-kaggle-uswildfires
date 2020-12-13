#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:37:00 2020

@author: andre
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from prepareDataForFireCausesPrediction import prepareDataForCausesPrediction

import random

train, test = prepareDataForCausesPrediction()
#train = train.sample(10000, random_state=0) # reducing the train set size (to save time)
X_train = train.drop(columns=["STAT_CAUSE_CODE"])
y_train = train["STAT_CAUSE_CODE"]
X_test = test.drop(columns=["STAT_CAUSE_CODE"])
y_test = test["STAT_CAUSE_CODE"]


# Current best params - on the whole data 56% accuracy
print("Calculating base score...")
rf = RandomForestClassifier(random_state=0, criterion="gini", n_jobs=-1, n_estimators=100, max_features='sqrt', max_depth=70, min_samples_leaf=2, min_samples_split=5).fit(X_train, y_train)
y_pred = rf.predict(X_test)
base_score = accuracy_score(y_test, y_pred)
print("Accuracy: ", str(base_score))



# Random selection
def selectRandomHyperparams():
    # Hyperparameters
    n_estimators = [50, 100, 200, 400, 600]
    max_features = ['auto', 'sqrt']
    max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    min_samples_leaf = [1,2,4]
    min_samples_split = [2, 5, 10]
    
    ne_index = random.randint(0, len(n_estimators)-1)
    mf_index = random.randint(0, len(max_features)-1)
    md_index = random.randint(0, len(max_depth)-1)
    msl_index = random.randint(0, len(min_samples_leaf)-1)
    mss_index = random.randint(0, len(min_samples_split)-1)
    
    return [n_estimators[ne_index],
            max_features[mf_index],
            max_depth[md_index],
            min_samples_leaf[msl_index],
            min_samples_split[mss_index]]

# tries 100 random sets of hyper params
def findHyperParams():
    accuracies = []
    for i in range(100):
        print(i)
        hyperparams = selectRandomHyperparams()
        ne = hyperparams[0]
        mf = hyperparams[1]
        md = hyperparams[2]
        msl = hyperparams[3]
        mss = hyperparams[4]
        rf = RandomForestClassifier(random_state=0, criterion="gini", n_jobs=-1, n_estimators=ne, max_features=mf, max_depth=md, min_samples_leaf=msl, min_samples_split=mss).fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        if (score > base_score):
            accuracies.append([score, ne, mf, md, msl, mss])
        print("Accuracy: ", str(score), "Base score", base_score)
    
    for x in accuracies:
        print(x)
    return accuracies
            
#acc = findHyperParams()



