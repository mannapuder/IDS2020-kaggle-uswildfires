import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from readSavedData import readfires

def fires_by_year():
    fire_counts=data["FIRE_YEAR"].value_counts().sort_index()
    X=fire_counts.index.values.reshape(-1,1)
    Y=fire_counts.values.reshape(-1,1)
    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    prediction = linear_regression.predict(X)

    plt.title("Number of wildfires in the US 1992-2015")
    plt.xlabel("Year")
    plt.ylabel("Number of fires")
    plt.bar(fire_counts.index, fire_counts.values)
    plt.plot(X, prediction, color='red')
    plt.savefig("./visuals/fires_by_year.png", bbox_inches="tight", dpi=800)
    plt.show()

def fires_by_causes():
    relevant_data = data.drop(["DISCOVERY_DOY", "DISCOVERY_DATE", "NEAREST_STATION", "STATION_DISTANCE", "LATITUDE", "LONGITUDE", "STATE"], axis=1)
    for i in range(1,14):
        one_cause_data = relevant_data[relevant_data["STAT_CAUSE_CODE"] == i]
        fire_counts=one_cause_data["FIRE_YEAR"].value_counts().sort_index()
        plt.plot(fire_counts.index, fire_counts.values, label=one_cause_data["STAT_CAUSE_DESCR"].values[0])
    plt.title("Wildfires in the US by causes 1992-2015")
    plt.xlabel("Year")
    plt.ylabel("Number of fires")
    plt.legend()
    plt.savefig("./visuals/fires_by_causes.png", dpi=800)
    plt.show()


data = readfires()

fires_by_year()
fires_by_causes()

