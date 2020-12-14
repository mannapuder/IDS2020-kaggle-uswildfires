from readSavedData import readfires
import plotly.graph_objects as go


def heatmap_by_states():
    fire_counts = data["STATE"].value_counts()

    fig = go.Figure(data=go.Choropleth(
        locations=fire_counts.index,
        z=fire_counts.values,
        locationmode="USA-states",
        colorbar_title="Number of wildfires",
        colorscale='reds'
    ))

    fig.update_layout(
        title_text="Wildfires by State 1992-2015",
        title_x=0.5,
        title_y=0.8,
        geo_scope='usa'
    )

    fig.write_image("./Visuals/heatmap_by_states.png", scale=2)
    fig.write_html("./Visuals/heatmap_by_states.html")


def heatmap_human_cause():
    relevant_data = data[["STATE", "STAT_CAUSE_CODE", "STAT_CAUSE_DESCR"]]
    cause_data = relevant_data[relevant_data["STAT_CAUSE_CODE"] != 1]
    fire_counts = cause_data["STATE"].value_counts()

    fig = go.Figure(data=go.Choropleth(
            locations=fire_counts.index,
            z=fire_counts.values,
            locationmode="USA-states",
            colorbar_title="Number of wildfires",
            colorscale='purples'
        ))

    fig.update_layout(
        title_text="Wildfires started from human cause by State 1992-2015",
        title_x=0.5,
        title_y=0.8,
        geo_scope='usa'
    )

    fig.write_image("./Visuals/heatmap_human_cause.png", scale=2)
    fig.write_html("./Visuals/heatmap_human_cause.html")


def heatmap_lightning():
    relevant_data = data[["STATE", "STAT_CAUSE_CODE", "STAT_CAUSE_DESCR"]]
    lightning_data = relevant_data[relevant_data["STAT_CAUSE_CODE"] == 1]
    fire_counts=lightning_data["STATE"].value_counts()

    fig = go.Figure(data=go.Choropleth(
        locations=fire_counts.index,
        z=fire_counts.values,
        locationmode="USA-states",
        colorbar_title="Number of wildfires",
        colorscale='teal'
    ))

    fig.update_layout(
        title_text="Wildfires caused by lightning by State 1992-2015",
        title_x=0.5,
        title_y=0.8,
        geo_scope='usa'
    )

    fig.write_image("./Visuals/heatmap_lightning.png", scale=2)
    fig.write_html("./Visuals/heatmap_lightning.html")


data = readfires()
heatmap_by_states()

