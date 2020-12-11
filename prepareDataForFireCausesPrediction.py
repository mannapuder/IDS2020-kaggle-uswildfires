#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:30:42 2020

@author: andre
"""

print("Importing libraries and data...")

import readSavedData as rd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler



def prepareDataForCausesPrediction(feature_scaling=False):
    pd.set_option("display.max_columns", 24)
    
    data = rd.merge()    
    
    print("Cleaning up the data...")
    
    # Removing the instances with too large distance
    data = data.drop(data[data["STATION_DISTANCE"] > 100].index)
    
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
    data = data.drop(data[data["STAT_CAUSE_CODE"] == 13].index)
    #data = data.drop(data[data["STAT_CAUSE_CODE"] == 9].index) 
    
    # Replacing all NaN values in WT03 and WV03 with zeros
    data['WT03'] = data['WT03'].fillna(0)
    data['WV03'] = data['WV03'].fillna(0)
    
    '''
    # Balancing the dataset by undersampling
    # Debris burning
    tmp = data[data["STAT_CAUSE_CODE"] == 5].sample(230000, random_state=0) # reduces the instances of debris burning to 230 000
    tmp2 = data[data["STAT_CAUSE_CODE"] != 5]
    data = pd.concat([tmp, tmp2])
    # Miscellaneous
    tmp = data[data["STAT_CAUSE_CODE"] == 9].sample(230000, random_state=0)
    tmp2 = data[data["STAT_CAUSE_CODE"] != 9]
    data = pd.concat([tmp, tmp2])
    '''
    
    # Feature scaling
    if (feature_scaling):
        print("Feature scaling...")
        minMaxScaler = MinMaxScaler()
        data[["FIRE_YEAR", "DISCOVERY_DOY", "LATITUDE", "LONGITUDE", "PRCP", "TMAX", "TMIN"]] = minMaxScaler.fit_transform(data[["FIRE_YEAR", "DISCOVERY_DOY", "LATITUDE", "LONGITUDE", "PRCP", "TMAX", "TMIN"]])
        
    
    data_dum = pd.get_dummies(data, columns=data.select_dtypes(include=["object"]).columns)
    print("Data shape: " + str(data_dum.shape))
    return data_dum


def testModels(normalize_features=False):

    # For testing
    print("Testing models...")
    sampleSize = 100000    
    
    data_dum = prepareDataForCausesPrediction(feature_scaling=normalize_features).sample(sampleSize) # for testing, otherwise some models run for hours
    X_train, X_test, y_train, y_test = train_test_split(data_dum.drop(columns=["STAT_CAUSE_CODE"]), data_dum["STAT_CAUSE_CODE"], test_size=0.2, random_state=0)
    
    print("Testing on sample with size " + str(sampleSize) + "...")
    
    print("Testing decision tree...")
    decisonTree = tree.DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train, y_train)
    y_pred = decisonTree.predict(X_test)
    print("Accuracy: ", str(accuracy_score(y_test, y_pred)))
    
    # n_jobs=-1 - run on all CPU cores
    
    #for i in range(5, 26, 5):
    i = 20
    print("Testing KNN ", str(i))
    knn = KNeighborsClassifier(n_neighbors = i, n_jobs=-1).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy: ", str(accuracy_score(y_test, y_pred)))
    
    print("Testing random forest...")
    rf = RandomForestClassifier(random_state=0, criterion="entropy", n_jobs=-1).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Accuracy: ", str(accuracy_score(y_test, y_pred)))
    
    #print("Testing gradient boosting...")
    #grad = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    #y_pred = grad.predict(X_test)
    #print("Accuracy: ", str(accuracy_score(y_test, y_pred))) 
    
    print("Testing StackingClassifier...")
    estimators = [('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier(n_neighbors = 20, n_jobs=-1))]
    stacking = StackingClassifier(estimators=estimators, n_jobs=-1).fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    print("Accuracy: ", str(accuracy_score(y_test, y_pred)))
        
    #svm_poly = SVC(kernel="poly", degree=2).fit(X_train, y_train)
    #y_pred = svm_poly.predict(X_test)
    #print("\nSVM poly")
    #print("Accuracy: ", str(accuracy_score(y_test, y_pred)))
    

#testModels(True)
