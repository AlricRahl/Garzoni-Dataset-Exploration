import pandas as pd

from tqdm import tqdm
from pandas_profiling import ProfileReport


def getReport(data, name):
    """Takes in the read data, creates a pandas report out of it

    :data: pandas dataframe
    TODO: Take custom configs to increase the speed of report generation
    """
    print(f"Profiling {name}...")
    profile = ProfileReport(data, title=f"{name.title()} Profile Report",
                            minimal=True)
    profile.to_file(f"../reports/{name.title()}_Profile_Report.html")


def readFromXlsx(filename, sheet_list=None):
    """Reads from an xlxs to a pandas dataframe, if the xlsx file has
    multiple sheets and a list is provided, returns a dictionary of frames
    instead

    """
    if sheet_list:
        sheets = dict()
        for sheet in tqdm(sheet_list):
            sheets[sheet] = pd.read_excel(filename, sheet_name=sheet)
        return sheets
    print("Reading file...")
    return pd.read_excel(filename)
