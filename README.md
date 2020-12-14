# IDS2020-kaggle-uswildfires
## The data
The data used was gathered from https://www.kaggle.com/rtatman/188-million-us-wildfires and https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ (files ghcnd_hcn.tar.gz and ghcnd-stations.txt). The selected date ranges and features were saved as csv files in a tar file in data/data.tar.xz.

### fires.csv:
Contains 1880465 rows of data with the following features:
1. FIRE_YEAR: the year the wildfire occurred.
1. DISCOVERY_DOY: the day of the year the wildfire was discovered.
1. DISCOVERY_DATE: the Julian date for the day the fire was discovered.
1. STAT_CAUSE_CODE: the numerical label for the statistical cause of the fire.
1. STAT_CAUSE_DESCR: the statistical cause of the fire.
1. LATITUDE: latitude (NAD83) for point location of the fire (decimal degrees).
1. LONGITUDE: longitude (NAD83) for point location of the fire (decimal degrees).
1. STATE: Two-letter code for the state in which the unit reporting the fire is located.
1. NEAREST_STATION: closest weather station's (from the HCN) id based on haversine distance.
1. STATION_DISTANCE: haversine distance in kilometers to the nearest_station.

#### Cause Code - Cause Description:
* 1 - Lightning
* 2 - Equipment Use
* 3 - Smoking
* 4 - Campfire
* 5 - Debris Burning
* 6 - Railroad
* 7 - Arson
* 8 - Children
* 9 - Miscellaneous
* 10 - Fireworks
* 11 - Powerline
* 12 - Structure
* 13 - Missing/Undefined
### stations.csv:
Information about the HCN station locations:
1. Station: station identification code.
1. Latitude: latitude of the station (in decimal degrees).
1. Longitude: longitude of the station (in decimal degrees).

### weather.csv:
Weather data from the HCN stations:
1. STATION: station identification code.
1. DATE: date of the measurements in format "yyyyMMdd".
1. PRCP: percipitation in tenths of millimeters.
1. PRCPQFLAG: failed quality check flag for percipitation.
1. TMAX: maximum temperature (tenths of degrees C).
1. TMAXQFLAG: failed quality check flag for maximum temperature.
1. TMIN: minimum temperature (tenths of degrees C).
1. TMINQFLAG: failed quality check flag for minimum temperature.
1. TAVG: average temperature (tenths of degrees C).
1. TAVGQFLAG: failed quality check flag for average temperature.
1. WT03: weather type: "Lightning".
1. WT03QFLAG: failed quality check flag for WT03.
1. WV03: weather in the vicinity: "Lightning".
1. WV03QFLAG: failed quality check flag for WV03.

Specific quality flag meanings can be found in https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt.
## Files:
### SelectFromOriginalData.py:
File for selecting relevant data from original data sources and saving them in data/data.tar.xz. Expects internet connection and files ghcnd_hcn.tar.gz and FPA_FOD_20170508.sqlite from the original data sources to be placed in the data/ folder. Running this file should not be necessary, as the selected data is already included in this repository.
Selects data according to following ideas:
1. From ghcnd_hcn.tar.gz:
   1. For each file in the archive, collect desired measurements (shown above) into groups according to the measurement date (while leaving out all measurements that do not fit into desired date range) and then write the results to data/weather.csv.
1. From FPA_FOD_20170508.sqlite select all desired fields (shown above aside from NEAREST_STATION and STATION_DISTANCE).
1. From ghcnd-stations.txt select Station id, Latitude and Longitude and write them to data/stations.csv.
1. Train a KNeighborsClassifier model with n_neighbors = 1 to predict a station id based on latitude and longitude useing haversine distance
1. For each fire, predict the nearest weather station using the model and save it to column NEAREST_STATION.
1. For each fire, calculate the haversine distance between it and its nearest station and save it to column STATION_DISTANCE.
1. Save the fires to data/fires.csv.
1. Pack weather.csv, stations.csv and fires.csv into a tar file with xz compression, then delete the csv files.

### readSavedData.py:
File for reading the csv files in data/data.tar.xz. Contains the following functions:
1. readweather(): reads the weather data in the weather.csv file to a pandas dataframe and converts the DATE field to pandas datetime format.
1. readstations(): reads the data in the stations.csv file to a pandas dataframe.
1. readfires(): reads the wildfire data in the fires.csv file to a pandas dataframe and converts the DISCOVERY_DATE field to pandas datetime format.
1. read(): returns the results of the three previous functions in the order of fires, weather, stations
1. merge(): reads and merges the weather and fire data based on the nearest station to the fire and the discovery date of the fire. Alternatively, instead of reading in the data, dataframes with the necessary information can be supplied (first weather, then fires).

### prepareDataForFireCausesPrediction.py:
File for cleaning data and testing a few models. Contains the following functions:
1. prepareDataForCausesPrediction(): returns train and test datasets that contain the merged data of the fires and weather datasets scaled with a MinMaxScaler, with train dataset having 80% of the cleaned data and the test dataset having the rest. Cleaning consists of:
   1. removing all rows that have a weather measurement that failed a quality check and removing the quality flag columns, 
   1. removing the TAVG column
   1. removing all fires with unknown cause
   1. replacing missing values for WT03 and WV03 with 0
   1. encoding state names with a LabelEncoder
1. testModels(): tests the following models and prints their accuracy scores: a DecisionTree with "entropy" criterion, a KNeighborsClassifier with 20 neighbors and a RandomForestClassifier with "entropy" criterion.
### fineTuneRandomForest.py:
File for optimizing the choice of hyperparameters for a random forest. Contains the following functions:
1. selectRandomHyperparams(): chooses randomly among a few given values for the following parameters: n_estimators, max_features, min_samples_leaf, min_samples_split
1. findHyperParams(): trains 100 random forests with random hyperparameters and reports the parameters along with the accuracy achieved.
### createModel.py:
File for creating the final model. Reads in and merges the fire and weather data, cleans and scales it following the steps given in prepareDataForFireCausesPrediction.py, trains a random forest (with "gini" criterion) with the best hyperparameters found with fineTuneRandomForest.py (n_estimators=100,  max_features='sqrt', max_depth=70, min_samples_leaf=2, min_samples_split=5) and writes the created model to file final_model.model, the used scaler to minMaxScaler.scaler and the used encoder to labelencoder.le, which can be read by using Python's pickle module's load() function.
### graphs.py:
File for creating visualisations of the fires dataset with pyplot. Contains the following functions (which all use the dataframe from the function readfires()):
1. fires_by_year(): Creates a bar plot of the number of wildfires per year and saves it to visuals/fires_by_year.png
1. fires_by_causes_and_years(): Creates a line graph of the number of fires per year grouped by causes and saves it to visuals/fires_by_causes_and_years.png
1. fires_by_causes(): Creates a bar plot of the number of fires per cause and saves it to visuals/fires_by_causes.png
### heatmaps.py:
File for creating geographical map type visualisations of the fires dataset. Contains the following functions (which all use the dataframe from the function readfires()):
1. heatmap_by_states(): Creates a heatmap of the number of fires in each state and saves it to visuals/heatmap_by_states.png.
1. heatmap_human_cause(): Creates a heatmap of the number of fires caused by human activity in each state and saves it to visuals/heatmap_human_cause.png
1. heatmap_lightning(): Creates a heatmap of the number of fires caused by lightning in each state and saves it to visuals/heatmap_lightning.png
All the heatmaps also have an interactive version in the form of an html file by the same name as the static image.
## Model creation:
1. Ensure the data folder contains data.tar.xz file
1. Run createModel.py
## Model usage:
### Loading in the model:
1. import the module "pickle"
1. use pickle.load("path_to_file") to load in the model from final_model.model, the scaler from inMaxScaler.scaler and the encoder from labelencoder.le and save them to variables
### Making predictions:
1. Ensure your data contains only the features FIRE_YEAR, DISCOVERY_DOY, LATITUDE, LONGITUDE, STATE, PRCP, TMAX, TMIN, WT03 and WV03 and that their format matches the original data formats. Prepare the data according to the following steps:
   1. Replace all missing values for WT03 and WV03 with 0.
   1. Remove all rows that still contain a missing value or figure out a reasonable replacement for the missing values.
   1. Scale the features FIRE_YEAR, DISCOVERY_DOY, LATITUDE, LONGITUDE, PRCP, TMAX and TMIN in that order using the transfomr function from the scaler from the file.
   1. Encode the feature STATE using the transform function from the encoder from the file.
1. Run the predict function of the model on your data
1. Refer to the code - description explanation given above to translate the predictions.
