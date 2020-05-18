import json
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

def readFile(filename, name):
    """This function reads the json file and reformat its for easy use

    :filename: path
    :name: string
    :returns: TODO

    """
    print(f'Reading {name}...')
    with open(filename, 'r') as myfile:
        data=myfile.read()
    
    # Load the json object and unravel the first layer
    json_d = json.loads(data)
    flattened = json_d[name]
    
    # Put it in a dataframe
    return pd.DataFrame(flattened)

def getReport(data, name):
    """Takes in the read data, creates a pandas report out of it

    :data: pandas dataframe
    TODO: Take custom configs to increase the speed of report generation
    """
    print(f'Profiling {name}...')
    profile = ProfileReport(data, title=f'{name.title()} Profile Report',
                            minimal = True)
    profile.to_file(f'../reports/{name.title()}_Profile_Report.html')

def amendNextract(frame, column):
    """ This function takes a dataframe, and a column with dict values in it 
    Then converts the column into a separate frame.
    TODO: We can have a key column as an input as well, so once we create 
    frames for everything, we can make one massive table to profile

    """
    tags = set()
    columnArr = frame[column].to_numpy()
    for item in columnArr:
        if type(item) is not float:
            tags = tags.union(set(item.keys()))
    
    null = dict()
    for item in tags:
        null[item] = None

    newFrame = list()
    for item in columnArr:
        if type(item) is not float:
            temp = null.copy()
            temp.update(item)
            newFrame.append(temp)
    return pd.DataFrame(newFrame)
