# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:36:36 2020

@author: kaarel
"""

import tarfile
import pandas as pd
import numpy as np

def read():
    return readfires(), readweather(), readstations()

def readweather():
    with tarfile.open("./data/data.tar.xz", "r:xz") as datafile:
        weather = pd.read_csv(datafile.extractfile("weather.csv"),
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
        weather.DATE = pd.to_datetime(weather.DATE, format="%Y%m%d")
        return weather


def readstations():
    with tarfile.open("./data/data.tar.xz", "r:xz") as datafile:
        return pd.read_csv(datafile.extractfile("stations.csv"))


def readfires():
    with tarfile.open("./data/data.tar.xz", "r:xz") as datafile:
        fires = pd.read_csv(datafile.extractfile("fires.csv"),dtype={"STAT_CAUSE_CODE":"uint8"})
        epoch = pd.to_datetime(0, unit='s').to_julian_date()
        fires.DISCOVERY_DATE = pd.to_datetime(fires.DISCOVERY_DATE-epoch, unit="D")
        return fires


def merge(weather=readweather(), fires=readfires()):
    return fires.merge(weather, how="left", left_on=["NEAREST_STATION","DISCOVERY_DATE"], right_on=["STATION","DATE"])

