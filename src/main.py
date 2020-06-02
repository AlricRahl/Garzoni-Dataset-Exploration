import myprofiler as mp

from tqdm import tqdm

if __name__ == "__main__":
    files = {
        'contracts': "../data/contracts_20200514_194518.xlsx",
        "locations": "../data/locations_20200520_143742.xlsx",
        "persons": "../data/persons_20200520_143122.xlsx",
        "personMentions": "../data/person_mentions_20200520_143017.xlsx",
        "personRelationships":
        "../data/person_relationships_20200520_143219.xlsx",
        "professions": "../data/professions_20200520_143812.xlsx",
        "categories": "../data/profession_categories_20200520_143818.xlsx",
    }
    
    print("Extracting sheets from contracts...")
    temp = mp.readFromXlsx("../data/contracts_20200520_142649.xlsx",
                           contracts_sheets)

    print("Reporting each contracts sheet...")
    for sheet_frame in tqdm(contracts_sheets):
        mp.getReport(temp[sheet_frame], "_".join(sheet_frame.split()))

    print("Extracting and Reporting Rest of the xlsx documents...")
    for name, file in tqdm(files.items()):
        temp = mp.readFromXlsx(file)
        mp.getReport(temp, name)
