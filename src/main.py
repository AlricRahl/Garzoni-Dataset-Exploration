import pandas as pd
import numpy as np
import profile


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

    # for name, file in files.items():
        # frame = profile.readFile(file, name)
        # profile.getReport(frame, name)
    
    pM = readFile(files['personMentions'], 'personMentions')

