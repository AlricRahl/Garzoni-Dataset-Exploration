import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from math import nan
import pickle as pkl

def saveExcel(df, filename, sheetname="Sheet1"):
    """Takes in a data frame, filename, and possibly sheetname, writes the
    dataframe in an excel file

    """
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheetname)
        writer.save()


def savePickle(df, filename):
    """Pickles frames"""
    with open("../data/pickles/"+filename, "wb") as file:
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
    except ValueError as e:
        return False


def plotHist(df, atype=int, **kwargs):
    try:
        plt.clf()
    except:
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
    else:
        return nan


def daysToHuman(time):
    """Takes in a number of days and turns into a readable form
    """
    years = time // 365
    time = years % 365
    months = time // 30
    days = time % 30
    return f"{years}/{months}/{days}"


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
