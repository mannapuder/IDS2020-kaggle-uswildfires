import pandas as pd
import tarfile
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def readfires():
    with tarfile.open("../data/data.tar.xz", "r:xz") as datafile:
        fires = pd.read_csv(datafile.extractfile("fires.csv"),dtype={"STAT_CAUSE_CODE":"uint8"})
        epoch = pd.to_datetime(0, unit='s').to_julian_date()
        fires.DISCOVERY_DATE = pd.to_datetime(fires.DISCOVERY_DATE-epoch, unit="D")
        return fires

data = readfires()

def fires_by_year():
    fire_counts=data["FIRE_YEAR"].value_counts().sort_index()
    X=fire_counts.index.values.reshape(-1,1)
    Y=fire_counts.values.reshape(-1,1)
    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    prediction = linear_regression.predict(X)

    plt.title("Number of fires by year")
    plt.xlabel("Years")
    plt.ylabel("Number of fires")
    plt.bar(fire_counts.index, fire_counts.values)
    plt.plot(X, prediction, color='red')
    plt.savefig("fires_by_year.png", bbox_inches="tight")
    plt.show()

def fires_by_causes():
    pass

