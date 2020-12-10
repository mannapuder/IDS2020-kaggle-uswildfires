#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:30:42 2020

@author: andre
"""

import readSavedData as rd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

print("Imports complete...")

def prepareDataForCausesPrediction():
    pd.set_option("display.max_columns", 24)
    
    data = rd.merge()    
    
    # Removing irrelevant columns
    # Maybe station distance is relevant?
    data = data.drop(columns=["STAT_CAUSE_DESCR", "NEAREST_STATION", "STATION_DISTANCE", "STATION", "DATE", "DISCOVERY_DATE"])
    
    
    # Removing the values that didn't past the quality checks
    data = data.drop(data[pd.isnull(data["PRCPQFLAG"]) == False].index)
    data = data.drop(data[pd.isnull(data["TMAXQFLAG"]) == False].index)
    data = data.drop(data[pd.isnull(data["TMINQFLAG"]) == False].index)
    data = data.drop(data[pd.isnull(data["TAVGQFLAG"]) == False].index)
    data = data.drop(data[pd.isnull(data["WT03QFLAG"]) == False].index)
    data = data.drop(data[pd.isnull(data["WV03QFLAG"]) == False].index)
    
    
    # Removing quality flag columns
    data = data.drop(columns=["PRCPQFLAG", "TMAXQFLAG", "TMINQFLAG", "TAVGQFLAG", "WT03QFLAG", "WV03QFLAG"])
    # Only 41 677 of 1.8 million rows have a value in column TAVG. So I removed the column entirely.
    data = data.drop(columns=["TAVG"])
    
    # Removing the rows with missing data
    data = data.dropna(subset=["FIRE_YEAR", "DISCOVERY_DOY", "STAT_CAUSE_CODE", "LATITUDE", "LONGITUDE", "STATE", "PRCP", "TMAX", "TMIN"])
    
    # Removing the rows where the cause of the fire is unknown
    data = data.drop(data[data["STAT_CAUSE_CODE"] == 13.0].index)
    
    # Replacing all NaN values in WT03 and WV03 with zeros
    data['WT03'] = data['WT03'].fillna(0)
    data['WV03'] = data['WV03'].fillna(0)
    
    data_dum = pd.get_dummies(data, columns=data.select_dtypes(include=["object"]).columns)
    return data_dum




# For testing
print("Testing models...")    

data_dum = prepareDataForCausesPrediction().head(10000) # for testing, otherwise some models run for hours
X_train, X_test, y_train, y_test = train_test_split(data_dum.drop(columns=["STAT_CAUSE_CODE"]), data_dum["STAT_CAUSE_CODE"], test_size=0.2, random_state=0)


decisonTree = tree.DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train, y_train)
y_pred = decisonTree.predict(X_test)
print("Decision tree")
print("Accuracy: ", str(accuracy_score(y_test, y_pred)))


knn = KNeighborsClassifier(n_neighbors = 3, n_jobs=3).fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN")
print("Accuracy: ", str(accuracy_score(y_test, y_pred)))


    
    
