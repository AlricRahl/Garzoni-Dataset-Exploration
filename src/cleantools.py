from datetime import datetime
from math import nan
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd


def savePickle(df, filename):
    """Pickles frames"""
    with open(filename, "wb") as file:
        pkl.dump(df, file)


def loadPickle(filename):
    """Loads from given pickle"""
    with open(filename, "rb") as file:
        return pkl.load(file)


def isInt(x):
    """Takes a string and checks if it can turn into int

    :x: TODO
    :returns: TODO

    """
    try:
        int(x)
        return True
    except ValueError:
        return False


def plotHist(df, atype=int, **kwargs):
    try:
        plt.clf()
    except Exception:
        print("Pass")
    df.astype(atype).plot.hist(**kwargs)
    plt.show()


def currencyMap(cur):
    """Takes in a currency string and converts it to soldi
    If the currency is not mapped, returns nan
    The currency information comes from pages:
    https://en.wikipedia.org/wiki/Venetian_grosso for ducato
    https://en.wikipedia.org/wiki/Venetian_lira for lire
    These sources are for an initial understanding, and can be replaced
    with more accurate ones if the research goes on this directions
    """
    if cur == "grz:Ducati":
        return 5
    elif cur == "grz:Soldi":
        return 1
    elif cur == "grz:Lire":
        return 20
    return nan


def dateToDays(x):
    """Takes a pd.Timestamp object, converts it to number of days"""
    x = np.array([int(a) for a in str(x.date()).split("-")])
    return x[0] * 365 + x[1] * 30 + x[2]


def normalizeTime(df, column):
    """Takes a dataframe and the name of the timestamp column,
    converts it to days, and normalizes it with the minimum present time"""
    df = df.copy()
    minTime = dateToDays(df[column].min())
    df[column] = df[column].apply(dateToDays).apply(lambda x: x - minTime + 1)
    return df


def daysToHuman(time):
    """Takes in a number of days and turns into a readable form
    """
    years = time // 365
    time = years % 365
    months = time // 30
    days = time % 30
    return f"{years}/{months}/{days}"


def betweenYears(
    df,
    s_year,
    date_column="Contract Date",
    s_month=1,
    s_day=1,
    e_year=1820,
    e_month=1,
    e_day=1,
):
    """Slices a dataframe for the given dates

    :df: TODO
    :start: TODO
    :end: TODO
    :returns: TODO

    """
    return df[
        (df[date_column] > datetime(s_year + 200, s_month, s_day))
        & (df[date_column] < datetime(e_year + 200, e_month, e_day))
    ]


def cleanColumns(df, to_go, eliminate_list=None):
    """This dataframe takes a dataframe and column names do delete
    Then deletes those columns. Furthermore, it takes can eliminate columns
    with given words as well.

    """
    titles = list(df.keys())
    cleaned = list()
    for title in titles:
        skip = False
        for head in to_go:
            if head == title:
                skip = True
                break
        for word in eliminate_list:
            if word in title:
                skip = True
                break
        if not skip:
            cleaned.append(title)
    return df[cleaned]


def hotEncode(df, attribute, group="Contract ID", operation="sum"):
    """Takes a df and an attribute, return dummy hot encodings

    :df: TODO
    :attribute: TODO
    :groups: TODO
    :returns: TODO

    """
    return (
        pd.concat([df[[group]], attribute.str.get_dummies()], axis=1)
        .groupby([group])
        .agg(operation)
        .reset_index()
    )
