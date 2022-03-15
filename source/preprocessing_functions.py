import numpy as np
import pandas as pd
import json
from scipy.stats import ttest_ind


# Loads files provided their path
# ===============================
def load_data(path):#,index):
    # Loads the data
    with open(path) as f:
        g = json.load(f)
    # Converts json dataset from dictionary to dataframe
    #print('Data loaded correctly')
    df = pd.DataFrame.from_dict(g)
    #df = df.set_index(index)
    return df


# Replaces string by NaN and delete the missing values
# ====================================================
def replace_delete_na(df,cols,char):
    df = df.copy()
    for el in cols:
        df[el] = df[el].replace(char,np.NaN)
    df.dropna(subset = cols, inplace=True)
    return df


# Checks for duplicated data and removes them
# ===========================================
def duplicated_data(df):
    # Copies the dataframe
    df = df.copy()
    # Rows containing duplicate data
    print("Removed ", df[df.duplicated()].shape[0], ' duplicate rows')
    # Returns a dataframe with the duplicated rows removed
    return df.drop_duplicates()




# Converts epoch time to datetime and sort by date
# I leave the format YY/mm/DD/HH:MM:SS since a priory we don't know the time scale of events
def to_datetime(df,var):
    # Copies the dataframe
    df = df.copy()
    if df[var].dtype!=int:
        df[var] = df[var].astype(int)
        
    #df[var] = pd.to_datetime(df[var], utc=True, format = "%Y%m%d",errors = 'coerce').dt.strftime('%Y-%m-%d')
    df[var] = pd.to_datetime(df[var], format = "%Y/%m/%d %H:%M:%S",errors = 'coerce').dt.strftime('%Y/%m/%d %H:%M:%S')
    df.sort_values(by=[var],inplace=True)
    df.reset_index(inplace=True,drop=True)
    # Returns the dataframe
    return df


# Checks for duplicated data
def duplicated_data(df):
    # Copies the dataframe
    df = df.copy()
    # Rows containing duplicate data
    print("Removed ", df[df.duplicated()].shape[0], ' duplicated rows.')
    # Returns a dataframe with the duplicated rows removed
    return df.drop_duplicates()


# Checks for columns with missing values (NaNs)
def check_missing_values(df,cols=None,axis=0):
    # Copies the dataframe
    df = df.copy()
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    result = missing_num.sort_values(by='missing_percent',ascending = False)
    # Returns a dataframe with columns with missing data as index and the number and percent of NaNs
    return result[result["missing_percent"]>0.0]



def id_to_road_lin(df,variable,rules):
    #Copies the dataframe
    df = df.copy()
    # Creates a new column with the type of distance
    newcol = []
    for el in df[variable]:
        if el[0] in rules:
            newcol.append('road')
        else:
            newcol.append('linear')
    df[variable] = newcol
    # Returns the dataframe
    return df


def outlier_removal(df,variables):
    #Copies the dataframe
    df = df.copy()
    #Filters the dataframe
    df_vars = df[variables]
    #Outliers removal
    Q1 = df_vars.quantile(0.25)
    Q3 = df_vars.quantile(0.75)
    IQR = Q3 - Q1
    df_vars = df_vars[~((df_vars < (Q1 - 1.5 * IQR)) | (df_vars > (Q3 + 1.5 * IQR))).any(axis=1)]
    df = df.iloc[df_vars.index]
    df.reset_index(inplace=True,drop=True)
    return df    


def t_student_test(x,y):
    stat, p = ttest_ind(x, y)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution.')
    else:
        print('Probably different distributions.')