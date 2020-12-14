#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 01:50:18 2020

Importing the model: model = pickle.load(open("final_model.model", "rb"))

@author: andre
"""
print("Importing data...")
import pandas as pd
import readSavedData as rd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


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


data_dum = pd.get_dummies(data, columns=data.select_dtypes(include=["object"]).columns)

 # Feature scaling
print("Feature scaling...")
minMaxScaler = MinMaxScaler()
data_dum[["FIRE_YEAR", "DISCOVERY_DOY", "LATITUDE", "LONGITUDE", "PRCP", "TMAX", "TMIN"]] = minMaxScaler.fit_transform(data_dum[["FIRE_YEAR", "DISCOVERY_DOY", "LATITUDE", "LONGITUDE", "PRCP", "TMAX", "TMIN"]])

print("Creating the model...")
data_x = data_dum.drop(columns=["STAT_CAUSE_CODE"])
data_y = data_dum["STAT_CAUSE_CODE"]
rf = RandomForestClassifier(criterion="gini", n_jobs=-1, n_estimators=100, max_features='sqrt', max_depth=70, min_samples_leaf=2, min_samples_split=5)
rf.fit(data_x, data_y)
pickle.dump(rf, open("final_model.model", "wb"))


