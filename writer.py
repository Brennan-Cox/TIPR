import os
import pandas as pd


def write_To_Excel(rows, cols, sheetName, path):
    """
    Writes a list of rows and columns to an excel file
    with a given sheet name and path
    if the file does not exist, it will be created
    if the file does exist, it will be appended to
    if the append fails, the file will be overwritten
    """
    # create df
    df = pd.DataFrame(rows, columns=cols)
    # check if path to file exists
    if not os.path.exists(path):
        df.to_excel(path, sheet_name=sheetName, index=False)
    else:
        try:
            with pd.ExcelWriter(path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                excel_df = pd.read_excel(path, sheet_name=sheetName)
                df = pd.concat([excel_df, df], ignore_index=True)
                df.to_excel(writer, sheet_name=sheetName, index=False)
        except Exception as e:
            print(e)
            print('Error writing to excel file')
            df.to_excel(path, sheet_name=sheetName, index=False)
