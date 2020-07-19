import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import geopandas as gpd

import cleantools as clnt


def plotMap(
    df, figsize, column="Count", empty_color="lightgrey", map_color_scale="Greens"
):
    """Plots the given column in a data frame on italy map"""
    # Assign objects for control
    fig, ax = plt.subplots(1, figsize=figsize, facecolor="lightblue")

    # Plot the values on empty map
    ax.set_title(
        "# of Apprentices in\nLog Scale",
        loc="center",
        color="grey",
        fontsize=20,
    )
    df.plot(ax=ax, color=empty_color, edgecolors="white")
    df.plot(column="Count", ax=ax, cmap=map_color_scale, edgecolors="black")

    # set an axis for the color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)

    # color bar
    vmax = df[column].max()
    mappable = plt.cm.ScalarMappable(
        cmap=map_color_scale, norm=plt.Normalize(vmin=0, vmax=vmax)
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=14)

    # Close axis for visible sea color
    ax.axis("off")
    plt.show()


def filterNmap(
    df,
    filterParam=None,
    s_year=1530,
    column="Count",
    empty_color="lightgrey",
    map_color_scale="Greens",
    figsize=(10,12),
    eu=False,
    **kwargs
):
    """Takes in a dataframe, a dictionary of columns and their desired values,
    plots the number of apprentices in provinces on an Italy map, using the
    data acquired by filtering the data frame accordingly"""
    if eu:
        gdf = gpd.read_file("../data/europe_map.shp")
        column_on_left = "Country"
        column_for_map = "NAME_ENGL"
    else:
        gdf = gpd.read_file("../data/italy_city_map.shp")
        column_on_left = "Apprentice Province"
        column_for_map = "NOME_PRO"
    df = clnt.filterFrame(df, filterParam, s_year=s_year, **kwargs)
    values = pd.DataFrame(
        df[column_on_left].value_counts().apply(np.log),
        columns=[column_on_left],
    )
    values = values.reset_index()
    values.columns = [column_on_left, column]
    df = gdf.merge(values, left_on=column_for_map, right_on=column_on_left, how="left")
    plotMap(
        df,
        figsize,
        column=column,
        empty_color=empty_color,
        map_color_scale=map_color_scale,
    )
