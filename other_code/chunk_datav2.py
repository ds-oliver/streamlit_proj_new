# Constants
DATA_TO_LOAD_PATH = 'data/chunked_data/original_results_players_data/'
CSV_PATH = 'data/chunked_data/csv_files/'
DB_PATH = 'data/chunked_data/db_files/'
FINAL_CSVS_PATH = 'data/data_out/csv_files/'
FINAL_DB_PATH = 'data/data_out/db_files/'
LEFT_MERGE_CSV_FILENAME = 'left_merge_df.csv'
ONLY_RESULTS_CSV_FILENAME = 'only_results_df.csv'

# Generate a timestamp string
timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create new database filenames with the timestamp
LEFT_MERGE_DB_FILENAME = f'left_merge_table_{timestamp_str}.db'
ONLY_RESULTS_DB_FILENAME = f'only_results_table_{timestamp_str}.db'

# Import statements and other dependencies
import datetime
import pandas as pd
import numpy as np
import uuid
import glob
import os
import sqlite3
import warnings
import time

# suppress warnings

warnings.filterwarnings("ignore", message="numpy.ufunc size changed.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.ndarray size changed.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*categorical_column.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*defaulting to pandas implementation.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*is deprecated and will be removed in a future version.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is deprecated and will be removed in a future version.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*is a deprecated alias for the builtin.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is a deprecated alias for the builtin.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*is deprecated.*", category=Warning)
warnings.filterwarnings("ignore", message=".*is private.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is private.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.*", category=UserWarning)

def test_database(db_path):
    """
    Tests the SQLite database by checking whether the connection can be established,
    whether specific tables exist, and whether data can be fetched from those tables.
    """
    conn = None
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path + RESULTS_DB_FILENAME)

        # Create a cursor
        cur = conn.cursor()

        # Execute a SELECT statement to fetch data from the database
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()

        # Check if tables exist
        if len(tables) == 0:
            print("There are no tables in the database. Something might have gone wrong.")
            return False

        # Fetch data from each table and check if it can be fetched
        for table in tables:
            try:
                cur.execute(f"SELECT * FROM {table[0]} LIMIT 5;")
                rows = cur.fetchall()
                if len(rows) == 0:
                    print(f"No data can be fetched from the table '{table[0]}'. Something might have gone wrong.")
                    return False
                else:
                    print(f"Data successfully fetched from the table '{table[0]}'.")

            except sqlite3.Error as e:
                print(f"An error occurred when trying to fetch data from the table '{table[0]}': {e}")
                return False

        print("The database seems to be operational.")

    except sqlite3.Error as e:
        print(f"An error occurred when trying to connect to the database: {e}")
        return False

    finally:
        if conn:
            conn.close()

    return True


LIST_OF_CSVS = ['players_all_seasons_data', 'df_2017_2018', 'df_2018_2019', 'df_2019_2020', 'df_2020_2021', 'df_2021_2022', 'df_2022_2023', 'df_1992_2016']

def load_data(data_to_load_path=DATA_TO_LOAD_PATH):
    list_of_csvs = LIST_OF_CSVS
    list_of_files = [data_to_load_path + csv + '.csv' for csv in list_of_csvs]

    dict_of_dfs = {csv: pd.read_csv(file) for csv, file in zip(list_of_csvs, list_of_files)}

    # print out the shape of each dataframe
    for csv, df in dict_of_dfs.items():
        print(f'{csv} shape: {df.shape}')

    return dict_of_dfs

def clean_results(results_df):
    drop_cols = ['score', 'match_report', 'notes']
    results_df = results_df.drop(columns=[col for col in drop_cols if col in results_df])

    desired_columns_order = ['season', 'gameweek', 'home_team', 'home_xg', 'away_xg', 'away_team', 'home_score', 'away_score', 'date', 'referee', 'venue', 'dayofweek',	'start_time', 'attendance']
    rest_of_columns = [col for col in results_df.columns if col not in desired_columns_order]
    results_df = results_df.reindex(desired_columns_order + rest_of_columns, axis=1)

    results_df['winning_team'] = np.where(results_df['home_score'] > results_df['away_score'], results_df['home_team'], np.where(results_df['home_score'] < results_df['away_score'], results_df['away_team'], 'draw'))
    results_df['losing_team'] = np.where(results_df['home_score'] < results_df['away_score'], results_df['home_team'], np.where(results_df['home_score'] > results_df['away_score'], results_df['away_team'], 'draw'))

    results_df['match_id'] = [uuid.uuid5(uuid.NAMESPACE_DNS, ''.join(map(str, row))) for row in zip(results_df['home_team'], results_df['away_team'], results_df['season'])]
    results_df['matchup_merge_key'] = [uuid.uuid5(uuid.NAMESPACE_DNS, ''.join(sorted(map(str, row)))) for row in zip(results_df['home_team'], results_df['away_team'])]
    results_df['season_merge_key'] = [uuid.uuid5(uuid.NAMESPACE_DNS, ''.join(sorted(map(str, row)))) for row in zip(results_df['home_team'], results_df['away_team'], results_df['season'])]

    # convert uuids to strings
    results_df['match_id'] = results_df['match_id'].astype(str)
    results_df['matchup_merge_key'] = results_df['matchup_merge_key'].astype(str)
    results_df['season_merge_key'] = results_df['season_merge_key'].astype(str)

    results_df['team'] = results_df['home_team']
    results_df['opponent'] = results_df['away_team']
    results_df['match_teams'] = ['_'.join(sorted(map(str, row))) for row in zip(results_df['team'], results_df['opponent'])]
    results_df['season_match_teams'] = results_df['match_teams'] + '_' + results_df['season'].astype(str)

    results_df = results_df.fillna(0)
    return results_df

# Clean the players dataframe
def clean_players(players_df):
    players_df = players_df.copy()

    # Fill missing values with 0 or 'None' based on the column type
    players_df = players_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('None'))

    # Drop unnecessary columns
    drop_cols = ['Unnamed: 0', 'shirtnumber']
    players_df = players_df.drop(columns=[col for col in drop_cols if col in players_df])

    # Rename columns
    players_df['year'] = players_df['season'].str[:4]
    players_df = players_df.rename(columns={'season': 'season_long', 'year': 'season', 'position_1': 'position'})

    # Create minutes_per90, match_teams, and season_match_teams columns
    players_df['minutes_per90'] = players_df['minutes'] / 90
    players_df['match_teams'] = ['_'.join(sorted(map(str, row))) for row in zip(players_df['team'], players_df['opponent'])]
    players_df['season_match_teams'] = players_df['match_teams'] + '_' + players_df['season'].astype(str)

    # Determine home_team and away_team based on 'home' column
    conditions = [
        (players_df['home'] == True),
        (players_df['home'] == False)
    ]
    choices_team = [players_df['team'], players_df['opponent']]
    choices_opponent = [players_df['opponent'], players_df['team']]
    players_df['home_team'] = np.select(conditions, choices_team)
    players_df['away_team'] = np.select(conditions, choices_opponent)

    # Create matchup_merge_key and season_merge_key columns
    players_df['matchup_merge_key'] = players_df.apply(lambda row: uuid.uuid5(uuid.NAMESPACE_DNS, ''.join(sorted([row['home_team'], row['away_team']]))) , axis=1)
    players_df['season_merge_key'] = players_df.apply(lambda row: uuid.uuid5(uuid.NAMESPACE_DNS, ''.join(sorted([row['home_team'], row['away_team'], str(row['season'])]))) , axis=1)

    # convert uuids to strings
    players_df['matchup_merge_key'] = players_df['matchup_merge_key'].astype(str)
    players_df['season_merge_key'] = players_df['season_merge_key'].astype(str)

    return players_df

# create a function that cuts the players_df and results_df into smaller dataframes based unique matchup_merge_key values
def cut_df(df, cols):
    """
    Summary: 
        cuts a dataframe based on unique values in cols

    Args:
        df (DataFrame): the dataframe to be cut
        cols (list): list of column names to cut the dataframe by

    Returns:
        dict: dictionary of dataframes
    """
    df_dict = {}
    for col in cols:
        unique_values = df[col].unique()
        for val in unique_values:
            df_dict[f"{col}_{val}"] = df[df[col] == val]
    return df_dict


# create a function that calculates per90 stats for each df in df_dict
def calculate_per90s(df_dict):
    """
    Summary: 
        calculates per90 stats for each df in df_dict for provided columns

    Args:
        df_dict (dict): dictionary of dataframes
        columns (list): list of columns for which to calculate per90 stats

    Returns:
        dict: dictionary of dataframes with additional per90 stats columns
    """
    # Iterate over each dataframe in the dictionary
    for key in df_dict:
        df = df_dict[key].copy()
        for col in df.select_dtypes(include=[np.number]).columns:  # Select only numeric columns
            if col != 'minutes':
                df.loc[:, f'{col}_per90'] = df[col] / (df['minutes'] / 90)
        
        df_dict[key] = df  # Save the updated dataframe back into the dictionary

    return df_dict

def save_as_csvs(df_dict, dataframe, csv_path=CSV_PATH, final_csvs_path=FINAL_CSVS_PATH):  
    """
    Summary: 
        saves each df in df_dict as a csv file and as a SQLite3 db file

    Args:
        df_dict (dict): dictionary of dataframes
        csv_path (str): directory to save csv files
        db_path (str): path to SQLite3 db file
        conn (Connection, optional): SQLite3 connection. Defaults to None.

    Returns:
        None
    """
    # Ensure directory for CSV files exists
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(final_csvs_path, exist_ok=True)

    # save df as a csv file
    # if database has player columns, save as players.csv else save as results.csv
    if 'player' in dataframe.columns:
        dataframe.to_csv(os.path.join(final_csvs_path, 'players.csv'), index=False)
    else:
        dataframe.to_csv(os.path.join(final_csvs_path, 'results.csv'), index=False)

    for key, df in df_dict.items():
        # Convert key to a string and replace spaces with underscores
        key = str(key).replace(" ", "_")  

        # Save as a csv file
        csv_file_path = os.path.join(csv_path, f"{key}.csv")
        df.to_csv(csv_file_path, index=False)
    

def save_as_dbs(players_dataframe, results_dataframe, db_path=DB_PATH):  
    """
    Saves each df in df_dict as a SQLite3 db file

    Args:
        df_dict (dict): dictionary of dataframes
        db_path (str): directory to save SQLite3 db files

    Returns:
        None
    """
    # Ensure directory for SQLite3 databases exists
    os.makedirs(db_path, exist_ok=True)

    # Save as a SQLite3 database
    players_dataframe.to_sql('players', sqlite3.connect(os.path.join(db_path, 'players.db')), index=False)
    results_dataframe.to_sql('results', sqlite3.connect(os.path.join(db_path, 'results.db')), index=False)



def clean_duplicate_columns(merged_df):
    """
    This function cleans the dataframe after a merge operation by removing duplicate columns.

    The function identifies duplicate columns based on their names. After a merge operation,
    pandas typically adds suffixes to columns with identical names (except the key used for merging).
    By default, these suffixes are '_x' and '_y'. This function removes these duplicate columns by
    keeping only the ones without these suffixes, and removing the ones with these suffixes.

    Args:
        merged_df (pd.DataFrame): The dataframe to be cleaned.

    Returns:
        pd.DataFrame: The cleaned dataframe with duplicate columns removed.
    """
    # Filter out the columns that end with '_x' or '_y' suffixes (as they are duplicates).
    # You might need to modify this if your pandas merge operation uses different suffixes.
    columns_to_keep = [col for col in merged_df.columns if not (col.endswith('_x') or col.endswith('_y'))]

    cleaned_df = merged_df[columns_to_keep]

    return cleaned_df

def main():
    
    # time the execution of the script
    start_time = time.time()

    # Create data directory if it doesn't exist
    os.makedirs(DATA_TO_LOAD_PATH, exist_ok=True)

    # Create final database directory if it doesn't exist
    os.makedirs(FINAL_DB_PATH, exist_ok=True)
    # Call load_data() function
    dict_of_dfs = load_data(DATA_TO_LOAD_PATH)

    # Create SQLite connection
    conn = sqlite3.connect(FINAL_DB_PATH + LEFT_MERGE_DB_FILENAME)
    print(f"--- {time.time() - start_time} seconds ---")
    try:
        # players_df is just the all_seasons_combined_df
        players_df = dict_of_dfs['players_all_seasons_data']

        # remove the all_seasons_combined_df from the dict_of_dfs
        dict_of_dfs.pop('players_all_seasons_data')

        # Concatenate the dataframes in the dictionary vertically
        all_results_df = pd.concat(dict_of_dfs.values(), axis=0)

        # Clean the dataframes
        all_results_df = clean_results(all_results_df)
        players_df = clean_players(players_df)

        only_results_df = all_results_df.copy()

        # set the index of the players_df and results_df to the matchup_merge_key column
        players_df.set_index('matchup_merge_key', inplace=True)
        all_results_df.set_index('matchup_merge_key', inplace=True)

        # merge on index
        print(f"Merging the dataframes...")
        left_merge_players_df = players_df.merge(all_results_df, left_index=True, right_index=True, how='left')
        print(f"Merge complete.")
        print(f"--- {time.time() - start_time} seconds ---")

        # reset the index and rename the index column
        left_merge_players_df.reset_index(inplace=True)
        left_merge_players_df.rename(columns={'index': 'matchup_merge_key'}, inplace=True)

        # rename match_teams_x to match_teams and season_match_teams_x to season_match_teams
        left_merge_players_df.rename(columns={'match_teams_x': 'match_teams', 'season_match_teams_x': 'season_match_teams'}, inplace=True)    
        left_merge_players_df = clean_duplicate_columns(left_merge_players_df)

        # Cut the dataframes into smaller ones
        # print fstring
        print(f"Chunking the dataframes...")
        left_merge_players_dict = cut_df(left_merge_players_df, ['match_teams', 'season_match_teams'])
        only_results_dict = cut_df(all_results_df, ['match_teams', 'season_match_teams'])
        print(f"Chunking complete.\n--- {time.time() - start_time} seconds ---")

        # Calculate per90 stats for each dataframe in the players_df_dict
        print(f"Calculating per90 stats...")
        left_merge_players_dict = calculate_per90s(left_merge_players_dict)
        print(f"Per90 stats calculated.\n--- {time.time() - start_time} seconds ---")

        # Save the dataframes
        print(f"Saving the dataframes...")
        save_as_csvs(left_merge_players_dict, left_merge_players_df, CSV_PATH, FINAL_CSVS_PATH)
        save_as_csvs(only_results_dict, all_results_df, CSV_PATH, FINAL_CSVS_PATH)

        save_as_dbs(left_merge_players_df, FINAL_DB_PATH, LEFT_MERGE_DB_FILENAME)
        save_as_dbs(only_results_df, FINAL_DB_PATH, ONLY_RESULTS_CSV_FILENAME)

        print(f"Dictionaries saved as CSVs & Dataframes saved as Databases saved.\n--- {time.time() - start_time} seconds ---")
        
    finally:
        conn.close()

    # Print the execution time
    print(f"Script executed in {time.time() - start_time} seconds.")

    return left_merge_players_df, only_results_df

if __name__ == "__main__":
    left_merge_players_df, only_results_df = main()
    print(left_merge_players_df.shape)
    print(left_merge_players_df.head())
    print(left_merge_players_df.info())
    print(left_merge_players_df.describe())
    print(left_merge_players_df.isnull().sum())
    print(left_merge_players_df['matchup_merge_key'].nunique())
    print(left_merge_players_df['matchup_merge_key'].value_counts())

