# -*- coding: utf-8 -*-
import os
import pandas as pd
import sqlite3
import tarfile
import sklearn.neighbors
import numpy as np

stationnames = []
desiredelements= (b"PRCP", b"TMAX", b"TMIN", b"TAVG", b"WT03", b"WV03")
with open("./data/weather.csv", "w") as resultfile:
    resultfile.write("STATION,DATE,PRCP,PRCPQFLAG,TMAX,TMAXQFLAG,TMIN,TMINQFLAG,TAVG,TAVGQFLAG,WT03,WT03QFLAG,WV03,WV03QFLAG\n")
    with tarfile.open("./data/ghcnd_hcn.tar.gz", "r:gz") as archive:
        for file in archive:
            if file.name in ("ghcnd_hcn", "ghcnd-version.txt"):
                continue
            station = file.name.split("/")[1].split(".")[0]
            stationnames.append(station)
            data = {}
            with archive.extractfile(file) as openfile:
                for line in openfile:
                    year = int(line[11:15])
                    if (1991<year<2016):
                        month = int(line[15:17])
                        yearmonth = year*100+month
                        element = line[17:21]
                        if element not in desiredelements:
                            continue
                        elementindex = desiredelements.index(element)
                        if (yearmonth not in data):
                            data[yearmonth] = [[None]*12 for i in range(31)]
                        for day in range(31):
                            elementvalue = int(line[(21+day*8):(26+day*8)])
                            elementflag = line[(27+day*8):(28+day*8)].decode()
                            data[yearmonth][day][elementindex*2] = None if elementvalue == -9999 else elementvalue
                            data[yearmonth][day][elementindex*2+1] = None if elementflag == " " else elementflag
            for item in data.items():
                yearmonth = item[0]
                for day in range(31):
                    if item[1][day].count(None) < 12:
                        resultfile.write(station+","+str(yearmonth*100+day+1))
                        for info in item[1][day]:
                            resultfile.write(",")
                            if not info == None:
                                resultfile.write(str(info))
                        resultfile.write("\n")
with sqlite3.connect("./data/FPA_FOD_20170508.sqlite") as conn:
    fires = pd.read_sql("select FIRE_YEAR, DISCOVERY_DOY, DISCOVERY_DATE, STAT_CAUSE_CODE, STAT_CAUSE_DESCR, LATITUDE, LONGITUDE, STATE from Fires", con=conn)
stations = pd.read_fwf("https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt", colspecs=[(0,11),(12,20),(21,30)], names=["Station","Latitude","Longitude"])
stations = stations[stations["Station"].isin(stationnames)]
stations.to_csv("./data/stations.csv", index=False)
knn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, metric="haversine").fit(np.radians(stations[["Latitude","Longitude"]]), stations.Station)
fires["NEAREST_STATION"] = knn1.predict(np.radians(fires[["LATITUDE", "LONGITUDE"]]))
fires["STATION_DISTANCE"] = fires.apply(
    lambda x: sklearn.neighbors.DistanceMetric.get_metric("haversine").pairwise(
        X=[[np.math.radians(x["LATITUDE"]), np.math.radians(x["LONGITUDE"])]],
        Y=[[np.math.radians(stations[stations.Station==x["NEAREST_STATION"]]["Latitude"]),np.math.radians(stations[stations.Station==x["NEAREST_STATION"]]["Longitude"])]])[0][0]* 6371, axis=1)
fires.to_csv("./data/fires.csv", index=False)
with tarfile.open("./data/data.tar.xz","w:xz") as result:
    result.add("./data/weather.csv", arcname="weather.csv")
    result.add("./data/fires.csv", arcname="fires.csv")
    result.add("./data/stations.csv", arcname="stations.csv")
os.remove("./data/weather.csv")
os.remove("./data/fires.csv")
os.remove("./data/stations.csv")