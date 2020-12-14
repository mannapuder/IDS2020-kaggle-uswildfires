import pandas as pd
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from readSavedData import readfires
import plotly.graph_objects as go

def heatmap_by_states():
    fire_counts = data["STATE"].value_counts()

    fig = go.Figure(data = go.Choropleth(
        locations=fire_counts.index,
        z = fire_counts.values,
        locationmode="USA-states",
        colorbar_title = "Number of wildfires",
        colorscale='reds'
    ))

    fig.update_layout(
        title_text = "Wildfires by State 1992-2015",
        title_x=0.5,
        title_y=0.8,
        geo_scope='usa'
    )

    fig.write_image("./Visuals/heatmap_by_states.png", scale=2)


def heatmap_by_coordinates():
    pass


data=readfires()

heatmap_by_states()