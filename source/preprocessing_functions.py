import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import ttest_ind
from sklearn.ensemble import IsolationForest


# Loads files provided their path
# ===============================
def load_data(path):
    """
    Loads data stored in a json file in path
    
    Args:
    path (string): Directory name where the json file is stored
    
    Returns:
    pd.DataFrame: DataFrame with the data.
    """
    # Loads the data
    with open(path) as f:
        g = json.load(f)
    # Converts json dataset from dictionary to dataframe
    #print('Data loaded correctly')
    df = pd.DataFrame.from_dict(g)
    #df = df.set_index(index)
    return df


# Checks for duplicated data and removes them
# ===========================================
def duplicated_data(df):
    """
    Checks for duplicated entries in a pandas dataframe
    
    Args:
    df (pd.DataFrame): A pandas dataframe
    
    Returns:
    pd.DataFrame: DataFrame with the duplicated data removed.
    """
    # Copies the dataframe
    df = df.copy()
    # Rows containing duplicate data
    print("Removed ", df[df.duplicated()].shape[0], ' duplicate rows')
    # Returns a dataframe with the duplicated rows removed
    return df.drop_duplicates()


# Replaces string by NaN and delete the missing values
# ====================================================
def replace_delete_na(df,cols,char):
    """
    Replaces string by NaN and delete the missing values.
    
    Args:
    df (pd.DataFrame): DataFrame with columns cols.
    cols (list:string): List of strings with the names of the columns to substitute by NaN.
    char (string): string to substitute by NaN
    
    Returns:
    pd.DataFrame: DataFrame with NaNs removed in cols.
    """
    df = df.copy()
    for el in cols:
        df[el] = df[el].replace(char,np.NaN)
    df.dropna(subset = cols, inplace=True)
    df.reset_index(drop=True)
    return df


# Checks for columns with missing values (NaNs)
# =============================================
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


# Converts epoch time to datetime and sort by date
# I leave the format YY/mm/DD/HH:MM:SS since a priory we don't know the time scale of events
# ==========================================================================================
def to_datetime(df, column_name):
    """
    Transforms a timestamp column to datetime and extracts components.
    
    Args:
    df (pd.DataFrame): DataFrame containing the timestamp column.
    column_name (str): The name of the column with timestamp values.
    
    Returns:
    pd.DataFrame: DataFrame with additional columns for year, month, day, hour, minute, and second.
    """
    df[column_name] = pd.to_datetime(df[column_name], unit='s')
    df['year'] = df[column_name].dt.year
    df['month'] = df[column_name].dt.month
    df['day'] = df[column_name].dt.day
    df['hour'] = df[column_name].dt.hour
    df['minute'] = df[column_name].dt.minute
    df['second'] = df[column_name].dt.second
    return df


# Creating the new distance type variable
# =======================================
def id_to_road_lin(df,variable,rules,name):    
    #Copies the dataframe
    df = df.copy()
    # Creates a new column with the type of distance
    newcol = []
    for el in df[variable]:
        if el[0] in rules:
            newcol.append('road')
        else:
            newcol.append('linear')
    # Creating the new column
    df[name] = newcol
    # Dropping the old column 
    df.drop(variable,axis=1,inplace=True)
    # Returns the dataframe
    return df


# Calculates the correlation coefficient between two variables
# ============================================================
def compute_correlations(df,x,y,z):
    # Get unique city labels from the 'city_id' column
    city_labels = df[z].unique()    
    # Dictionary to store the correlation results
    correlation_results = {}
    # Loop through each unique city label
    for city in city_labels:
        # Filter the DataFrame for the current city
        city_data = df[df[z] == city]        
        # Compute correlation coefficient between duration and distance
        correlation = city_data[x].corr(city_data[y])        
        # Store the result in the dictionary
        correlation_results[city] = correlation    
    return correlation_results


# Removes outliers
# ================
def remove_outliers(df, variables,contamination):
    iso = IsolationForest(contamination=contamination)  # 50% contamination rate
    yhat = iso.fit_predict(df[variables])
    mask = yhat != -1
    df_filtered = df[mask]
    return df_filtered


#Plot Distributions
# =================
def boxplot_distributions(df,x,y,z,u):
    # Get the unique city labels from 'city_id'
    cities = df[u].unique()
    
    # Loop through each unique city
    for city in cities:
        # Filter the DataFrame by the current city
        city_data = df[df[u] == city]        
        # Create subplots for duration and distance
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Boxplot for Duration
        sns.boxplot(x=z, y=x, data=city_data, ax=axs[0])
        axs[0].set_title(f'Duration in {city}')
        axs[0].set_xlabel('Distance Type')
        axs[0].set_ylabel('Duration')
        
        # Boxplot for Distance
        sns.boxplot(x=z, y=y, data=city_data, ax=axs[1])
        axs[1].set_title(f'Distance in {city}')
        axs[1].set_xlabel('Distance Type')
        axs[1].set_ylabel('Distance')
        
        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()

# PSI and Uplift factor
# =====================
def psi_and_uplift_per_city(df, value_column, city_column, type_column, distance_types=['linear', 'road'], bins=10):
    """
    Calculate PSI and uplift for each unique city in the dataset.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - value_column: The column on which to compute PSI and uplift (e.g., 'distance' or 'duration').
    - city_column: The name of the column that identifies the city (e.g., 'city_id').
    - type_column: The name of the column that specifies the type (e.g., 'distance_type').
    - distance_types: The two types of distance to compare (default ['linear', 'road']).
    - bins: Number of bins to use for PSI calculation.

    Returns:
    - A DataFrame with city_id, PSI, and uplift values for each city.
    """
    
    def calculate_psi(expected, actual, bins):
        """Calculate the Population Stability Index (PSI) for two distributions."""
        expected_hist, bin_edges = np.histogram(expected, bins=bins)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
        
        # Avoid division by zero and log of zero by adding a small value (epsilon)
        epsilon = 1e-10
        expected_percent = (expected_hist / len(expected)) + epsilon
        actual_percent = (actual_hist / len(actual)) + epsilon
        
        psi = np.sum((expected_percent - actual_percent) * np.log(expected_percent / actual_percent))
        return psi

    # Get the unique city labels from the specified city column
    cities = df[city_column].unique()
    
    # List to store results
    results = []

    # Loop through each unique city
    for city in cities:
        # Filter the DataFrame by the current city
        city_data = df[df[city_column] == city]
        
        # Separate data into two groups: linear and road based on type_column
        linear_data = city_data[city_data[type_column] == distance_types[0]][value_column]
        road_data = city_data[city_data[type_column] == distance_types[1]][value_column]
        
        # Check if we have enough data for both groups to calculate PSI and uplift
        if len(linear_data) == 0 or len(road_data) == 0:
            # If either group is empty, skip the calculation for this city
            continue
        
        # Calculate PSI between the two distributions
        psi_value = calculate_psi(linear_data, road_data, bins)
        
        # Calculate uplift (difference in means between linear and road)
        uplift_value = (linear_data.median() - road_data.median())/linear_data.median()
        
        # Append the result to the list
        results.append({
            city_column: city,
            'PSI': psi_value,
            'Uplift': uplift_value
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# T-Student Test
# ==============
def t_student_test(x,y):
    stat, p = ttest_ind(x, y)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution.')
    else:
        print('Probably different distributions.')
        
        
# Most performant vehicles
# ========================
def vehicle_performance(df):
    # Group by vehicle_id and calculate metrics
    vehicle_stats = df.groupby('vehicle_id').agg(
        total_rides=('vehicle_id', 'size'),     # Count the number of rides per vehicle
        avg_distance=('distance', 'mean'),      # Calculate the average distance per vehicle
        avg_duration=('duration', 'mean')       # Calculate the average duration per vehicle
    ).reset_index()

    # Find the vehicles with most and least rides
    most_rides_vehicle = vehicle_stats.loc[vehicle_stats['total_rides'].idxmax()]
    least_rides_vehicle = vehicle_stats.loc[vehicle_stats['total_rides'].idxmin()]

    # Find the vehicles with highest and lowest average distance and duration
    best_performing_vehicle = vehicle_stats.loc[
        (vehicle_stats['avg_distance'].idxmax()) & (vehicle_stats['avg_duration'].idxmax())
    ]
    worst_performing_vehicle = vehicle_stats.loc[
        (vehicle_stats['avg_distance'].idxmin()) & (vehicle_stats['avg_duration'].idxmin())
    ]

    # Create a result dictionary to summarize
    result = {
        'Most Rides': most_rides_vehicle.to_dict(),
        'Least Rides': least_rides_vehicle.to_dict(),
        'Best Performing Vehicle': best_performing_vehicle.to_dict(),
        'Worst Performing Vehicle': worst_performing_vehicle.to_dict()
    }

    return result