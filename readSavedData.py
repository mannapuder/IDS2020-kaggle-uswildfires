# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:36:36 2020

@author: kaarel
"""

import tarfile
import pandas as pd
import numpy as np

def read(file=None):
    with tarfile.open("./data/data.tar.xz", "r:xz") as datafile:
        if file=="stations":
            return pd.read_csv(datafile.extractfile("stations.csv"))
        if file == "weather":
            return pd.read_csv(datafile.extractfile("weather.csv"),
                          dtype={
                              "STATION":np.object,
                              "DATE":"Int64",
                              "PRCP":"Int64",
                              "PRCPQFLAG":np.object,
                              "TMAX":"Int64",
                              "TMAXQFLAG":np.object,
                              "TMIN":"Int64",
                              "TMINQFLAG":np.object,
                              "TAVG":"Int64",
                              "TAVGQFLAG":np.object,
                              "WT03":"Int64",
                              "WT03QFLAG":np.object,
                              "WV03":"Int64",
                              "WV03QFLAG":np.object
                              })
        if file=="fires":
            return pd.read_csv(datafile.extractfile("fires.csv"))
        return pd.read_csv(datafile.extractfile("fires.csv")), pd.read_csv(datafile.extractfile("weather.csv"),
                          dtype={
                              "STATION":np.object,
                              "DATE":"Int64",
                              "PRCP":"Int64",
                              "PRCPQFLAG":np.object,
                              "TMAX":"Int64",
                              "TMAXQFLAG":np.object,
                              "TMIN":"Int64",
                              "TMINQFLAG":np.object,
                              "TAVG":"Int64",
                              "TAVGQFLAG":np.object,
                              "WT03":"Int64",
                              "WT03QFLAG":np.object,
                              "WV03":"Int64",
                              "WV03QFLAG":np.object
                              }), pd.read_csv(datafile.extractfile("stations.csv"))
        