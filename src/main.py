import pandas as pd
import numpy as np
from myprofiler import readFile, getReport, amendNextract

if __name__ == "__main__":
    files = {
        'contracts': "../data/contracts_20200514_194518.json",
        'locations': "../data/locations_20200514_203113.json",
        'persons': "../data/persons_20200514_201258.json",
        'personMentions': "../data/person_mentions_20200514_194949.json",
        'personRelationships': "../data/person_relationships_20200514_201308.json",
        'professions': "../data/professions_20200514_203119.json",
        'categories': "../data/profession_categories_20200514_203123.json"
    }
    # Here we create the initial reports
    # for name, file in files.items():
        # frame = readFile(file, name)
        # getReport(frame, name)
    
    # After observing missing parts in reports, we extract some columns as
    # new dataframes and analyze them separately
    # These are the columns with dictionaries
    dict_variables = {
        'persons': ['relationships'],
        'professions': None,
        'contracts': ['mentions'],
        'personRelationships': None,
        'personMentions': ['name', 'entity', 'professions', 'workshop',
                           'geoOrigin', 'chargeLocation', 'residence'],
        'locations': None 
    }

    # Here we decide only these are worth reporting
    pM = readFile(files['personMentions'], 'personMentions')
    for item in ['workshop', 'name']:
        frame = amendNextract(pM, item)
        getReport(frame, item)
