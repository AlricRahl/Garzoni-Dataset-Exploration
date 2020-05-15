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
    :returns: TODO

    """
    print(f'Profiling {name}...')
    profile = ProfileReport(data, title=f'{name.title()} Profile Report',
                            minimal = True)
    profile.to_file(f'../reports/{name.title()}_Profile_Report.html')

if __name__ == "__main__":
    files = {
        'contracts': "../data/contracts_20200514_194518.json",
        'locations': "../data/locations_20200514_203113.json",
        'persons': "../data/persons_20200514_201258.json",
        'mentions': "../data/person_mentions_20200514_194949.json",
        'relationships': "../data/person_relationships_20200514_201308.json",
        'professions': "../data/professions_20200514_203119.json",
        'categories': "../data/profession_categories_20200514_203123.json"
    }

    for name, file in files.items():
        frame = readFile(file, name)
        getReport(frame, name)
