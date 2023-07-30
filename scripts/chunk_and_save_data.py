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
import datetime
import pickle
# At the top of chunk_and_save_data.py
import sys
sys.path.append(
    '/Users/hogan/dev/streamlit_proj_new/helper_functions')
from log_it import set_up_logs, log_start_of_script, log_end_of_script, log_start_of_function, log_end_of_function, log_start_of_app, log_end_of_app, log_dataframe_details, log_specific_info_message, log_dict_contents

# Constants
DATA_IN_PATH = 'data/data_in/original_results_players_data/'

PLAYERS_CHUNKED_CSVS_PATH = 'data/data_out/chunked_data/players/'
RESULTS_CHUNKED_CSVS_PATH = 'data/data_out/chunked_data/results/'

FINAL_DICTS_PATH = 'data/data_out/final_data/pickle_files/'
FINAL_CSVS_PATH = 'data/data_out/final_data/csv_files/'

FINAL_DB_PATH = "data/data_out/final_data/db_files/"

# make directory if it doesn't exist
if not os.path.exists(FINAL_DB_PATH):
    os.makedirs(FINAL_DB_PATH)

# make directory if it doesn't exist
if not os.path.exists(FINAL_CSVS_PATH):
    os.makedirs(FINAL_CSVS_PATH)

# make directory if it doesn't exist
if not os.path.exists(PLAYERS_CHUNKED_CSVS_PATH):
    os.makedirs(PLAYERS_CHUNKED_CSVS_PATH)

# make directory if it doesn't exist
if not os.path.exists(RESULTS_CHUNKED_CSVS_PATH):
    os.makedirs(RESULTS_CHUNKED_CSVS_PATH)


# Generate a timestamp string
timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

LEFT_MERGE_CSV_FILENAME = f'left_merge_df_{timestamp_str}.csv'
ONLY_RESULTS_CSV_FILENAME = f'only_results_df_{timestamp_str}.csv'
# Create new database filenames with the timestamp
LEFT_MERGE_DB_FILENAME = f'left_merge_table_{timestamp_str}.db'
ONLY_RESULTS_DB_FILENAME = f'only_results_table_{timestamp_str}.db'

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

# set up logs
log_file_path = set_up_logs()

# log start of app
app_start_time = log_start_of_app(log_file_path)

# log start of script
script_start_time, script_end_time = log_start_of_script(log_file_path)

LIST_OF_CSVS = ['players_all_seasons_data', 'df_2017_2018', 'df_2018_2019', 'df_2019_2020', 'df_2020_2021', 'df_2021_2022', 'df_2022_2023', 'df_1992_2016']

def load_data(data_in_path=DATA_IN_PATH):
    list_of_csvs = LIST_OF_CSVS
    list_of_files = [data_in_path + csv + '.csv' for csv in list_of_csvs]

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

    results_df['team'] = results_df['home_team']
    results_df['opponent'] = results_df['away_team']
    results_df['match_teams'] = ['_'.join(sorted(map(str, row))) for row in zip(results_df['team'], results_df['opponent'])]
    results_df['season_match_teams'] = results_df['match_teams'] + '_' + results_df['season'].astype(str)

    # strip whitespace from match_teams column and season_match_teams column
    results_df['match_teams'] = results_df['match_teams'].str.replace(' ', '_')
    results_df['season_match_teams'] = results_df['season_match_teams'].replace(' ', '_')

    # convert to string
    results_df['match_teams'] = results_df['match_teams'].astype(str)
    results_df['season_match_teams'] = results_df['season_match_teams'].astype(str)

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

    # make sure there is no whitespace in match_teams and season_match_teams
    players_df['match_teams'] = players_df['match_teams'].str.replace(' ', '_')
    players_df['season_match_teams'] = players_df['season_match_teams'].str.replace(' ', '_')

    # convert to string
    players_df['match_teams'] = players_df['match_teams'].astype(str)
    players_df['season_match_teams'] = players_df['season_match_teams'].astype(str)

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

def save_csvs(dataframe, dataframe_name):
    # determine the csv filename based on whether 'player' is in the dataframe columns
    csv_filename = 'players.csv' if 'player' in dataframe.columns else 'results.csv'
    
    # save the dataframe to the final_csvs_path
    dataframe.to_csv(FINAL_CSVS_PATH + csv_filename, index=False)
    print(f'Saved {dataframe_name} as {csv_filename} in {FINAL_CSVS_PATH}')


def save_dicts(dictionary, dictionary_name):
    # check whether the dictionary_name contains 'players' or 'results'
    if 'players' in dictionary_name:
        csvs_path = PLAYERS_CHUNKED_CSVS_PATH
        csv_filename_prefix = 'players_'
    elif 'results' in dictionary_name:
        csvs_path = RESULTS_CHUNKED_CSVS_PATH
        csv_filename_prefix = 'results_'
    else:
        print('Error: Invalid dictionary_name.')
        return

    # save each dataframe in the dictionary to its own csv file
    for key, df in dictionary.items():
        csv_filename = csv_filename_prefix + f"{key}.csv"
        df.to_csv(csvs_path + csv_filename, index=False)
        print(f'Saved dataframe with key {key} from {dictionary_name} as {csv_filename} in {csvs_path}')

    # save the entire dictionary as a pickle file
    pickle_filename = dictionary_name + '.pkl'
    with open(FINAL_DICTS_PATH + pickle_filename, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f'Saved {dictionary_name} as {pickle_filename} in {FINAL_DICTS_PATH}')

def load_players_data_from_csvs(players_chunked_csvs_path=PLAYERS_CHUNKED_CSVS_PATH):
    """
    Summary: 
        loads data from csv files into a dictionary of dataframes

    Args:
        csv_path (str): directory to load csv files

    Returns:
        dict: dictionary of dataframes
    """
    # Ensure directory for CSV files exists
    os.makedirs(players_chunked_csvs_path, exist_ok=True)

    # Create an empty dictionary to store dataframes
    df_dict = {}

    # Iterate over each file in the directory
    for file in os.listdir(players_chunked_csvs_path):
        # Load the CSV file into a dataframe
        df = pd.read_csv(os.path.join(players_chunked_csvs_path, file))

        # Convert the filename to a key and save the dataframe into the dictionary
        key = file.replace(".csv", "")
        df_dict[key] = df

    return df_dict


def save_as_dbs(players_dataframe, results_dataframe, FINAL_DB_PATH=FINAL_DB_PATH):  
    """
    Saves each df in df_dict as a SQLite3 db file

    Args:
        df_dict (dict): dictionary of dataframes
        FINAL_DB_PATH (str): directory to save SQLite3 db files

    Returns:
        None
    """
    # Ensure directory for SQLite3 databases exists
    os.makedirs(FINAL_DB_PATH, exist_ok=True)

    # Save as a SQLite3 database
    players_dataframe.to_sql('players', sqlite3.connect(os.path.join(FINAL_DB_PATH, 'players.db')), if_exists='replace', index=False)
    results_dataframe.to_sql('results', sqlite3.connect(os.path.join(FINAL_DB_PATH, 'results.db')), if_exists='replace', index=False)

def load_data_from_db(FINAL_DB_PATH=FINAL_DB_PATH):
    """
    Loads data from SQLite3 databases

    Args:
        FINAL_DB_PATH (str): directory to save SQLite3 db files

    Returns:
        players_df (DataFrame): dataframe of players data
        results_df (DataFrame): dataframe of results data
    """

    # Load data from SQLite databases
    conn_players = sqlite3.connect(os.path.join(FINAL_DB_PATH, 'players.db'))
    cursor = conn_players.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())
    players_table = pd.read_sql_query("SELECT * FROM players", conn_players)

    conn_results = sqlite3.connect(os.path.join(FINAL_DB_PATH, 'results.db'))
    results_table = pd.read_sql_query("SELECT * FROM results", conn_results)

    return players_table, results_table

def filter_into_smaller_tables(players_table, results_table):
    """
    Summary:
        filters players_table and results_table into smaller tables based on unique matchup_merge_key values

    Args:
        players_table (DataFrame): dataframe of players data
        results_table (DataFrame): dataframe of results data

    Returns:
        team_specific_players_dict (dict): dictionary of dataframes of players data filtered by team
        team_specific_results_dict (dict): dictionary of dataframes of results data filtered by team
        player_specific_players_dict (dict): dictionary of dataframes of players data filtered by player
    """
        # Here you need to specify the logic to filter the dataframes based on unique values of `matchup_merge_key`
    team_specific_players_dict = {}
    team_specific_results_dict = {}
    player_specific_players_dict = {}

    # Assuming matchup_merge_key is a column in your dataframes
    unique_keys = players_table['matchup_merge_key'].unique()

    for key in unique_keys:
        # Filter players_table for each unique key and store in dictionary
        team_specific_players_dict[key] = players_table[players_table['matchup_merge_key'] == key]

        # Filter results_table for each unique key and store in dictionary
        team_specific_results_dict[key] = results_table[results_table['matchup_merge_key'] == key]

        # Filter players_table for each unique player and store in dictionary
        player_specific_players_dict[key] = players_table[players_table['player'] == key]

    return team_specific_players_dict, team_specific_results_dict, player_specific_players_dict


def clean_duplicate_columns(merged_df):
    # Create sets of '_x' and '_y' columns
    cols_x = {col for col in merged_df.columns if col.endswith('_x')}
    cols_y = {col for col in merged_df.columns if col.endswith('_y')}

    # Find overlapping column names (without suffixes)
    overlapping_cols = {col[:-2] for col in cols_x & cols_y}

    # Rename '_x' columns and drop '_y' columns for overlapping names only
    merged_df = merged_df.rename(columns={col: col[:-2] for col in cols_x if col[:-2] in overlapping_cols})
    merged_df = merged_df.drop(columns=[col for col in cols_y if col[:-2] in overlapping_cols])

    return merged_df


def main():
    
    # log start of main() function
    main_function_start_time = log_start_of_function('main()')
    
    # time the execution of the script
    start_time = time.time()

    # Create data directory if it doesn't exist
    os.makedirs(DATA_IN_PATH, exist_ok=True)

    # Create final database directory if it doesn't exist
    os.makedirs(FINAL_DB_PATH, exist_ok=True)
    # Call load_data() function
    dict_of_dfs = load_data(DATA_IN_PATH)

    # Create SQLite connection
    conn = sqlite3.connect(FINAL_DB_PATH + LEFT_MERGE_DB_FILENAME)
    print(f"Opened database successfully")
    print(f"--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start ---")

    try:
        # players_df is just the all_seasons_combined_df
        players_df = dict_of_dfs['players_all_seasons_data']

        # remove the all_seasons_combined_df from the dict_of_dfs
        dict_of_dfs.pop('players_all_seasons_data')

        # Concatenate the dataframes in the dictionary vertically
        only_results_df = pd.concat(dict_of_dfs.values(), axis=0)

        # pop df_1992_2016
        dict_of_dfs.pop('df_1992_2016')

        # create matching results df by concatenating 'df_2017_2018', 'df_2018_2019', 'df_2019_2020', 'df_2020_2021', 'df_2021_2022', 'df_2022_2023'
        matching_results_df = pd.concat(dict_of_dfs.values(), axis=0)
        print(f"matching_results_df shape: {matching_results_df.shape}")

        # Clean the dataframes
        clean_results_start_time = log_start_of_function('clean_results')

        only_results_df = clean_results(only_results_df)

        # log dataframes details
        log_dataframe_details('only_results_df', only_results_df)      

        matching_results_df = clean_results(matching_results_df)
        
        # log dataframes details
        log_dataframe_details('matching_results_df', matching_results_df)

        players_df = clean_players(players_df)

        # log dataframes details
        log_dataframe_details('players_df', players_df)

        log_end_of_function('clean_results', clean_results_start_time, app_start_time)

        # # set the index of the players_df and results_df to the matchup_merge_key column
        # players_df.set_index('match_teams', inplace=True)
        # matching_results_df.set_index('match_teams', inplace=True)

        # Drop rows with missing values
        matching_results_df.dropna(subset=['home_team', 'away_team', 'gameweek', 'season'], inplace=True)

        # Convert all values to string
        matching_results_df['home_team'] = matching_results_df['home_team'].astype(str)
        matching_results_df['away_team'] = matching_results_df['away_team'].astype(str)
        matching_results_df['gameweek'] = matching_results_df['gameweek'].astype(str)
        matching_results_df['season'] = matching_results_df['season'].astype(str)

        # We need to process the 'season' column a bit differently, as it requires additional manipulation
        # Remove the suffix from the 'season' column (e.g., '2017-2018' becomes '2017'), then convert to string
        matching_results_df['season'] = matching_results_df['season'].str.slice(0, 4)

        # Create the matchup_merge_key
        matching_results_df['matchup_merge_key'] = matching_results_df.apply(lambda row: '_'.join(sorted([row['home_team'], row['away_team']]) + [row['gameweek'], row['season']]), axis=1)

        # make sure all whitespace is removed from the matchup_merge_key column
        matching_results_df['matchup_merge_key'] = matching_results_df['matchup_merge_key'].str.replace(' ', '_')

        # Drop rows with missing values
        players_df.dropna(subset=['team', 'opponent', 'gameweek', 'season'], inplace=True)

        # Convert all values to string
        players_df['team'] = players_df['team'].astype(str)
        players_df['opponent'] = players_df['opponent'].astype(str)
        players_df['gameweek'] = players_df['gameweek'].astype(str)
        players_df['season'] = players_df['season'].astype(str)

        # We need to process the 'season' column a bit differently, as it requires additional manipulation
        # Remove the suffix from the 'season' column (e.g., '2017-2018' becomes '2017'), then convert to string
        players_df['season'] = players_df['season'].str.slice(0, 4)

        # Create the matchup_merge_key
        players_df['matchup_merge_key'] = players_df.apply(lambda row: '_'.join(sorted([row['team'], row['opponent']]) + [row['gameweek'], row['season']]), axis=1)

        # make sure all whitespace is removed from the matchup_merge_key column
        players_df['matchup_merge_key'] = players_df['matchup_merge_key'].str.replace(' ', '_')

        # merge on index
        print(f"Merging the dataframes...")
        merge_start_of_function = log_start_of_function('merge')

        # Now we can merge the two dataframes on the 'matchup_merge_key' column
        left_merge_players_df = pd.merge(players_df, matching_results_df, on='matchup_merge_key', how='left')


        # Finally, let's drop the 'matchup_merge_key' column as it is no longer needed
        left_merge_players_df = left_merge_players_df.drop('matchup_merge_key', axis=1)

        # log specific message
        print(f"Merge of players_df and matching_results_df to left_merge_players_df complete.")

        log_end_of_function('merge', merge_start_of_function, app_start_time)

        # log dataframes details
        log_dataframe_details('left_merge_players_df', left_merge_players_df)

        print(f"Merge complete.")
        print(f"--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start ---")

        # # reset the index and rename the index column
        # left_merge_players_df.reset_index(inplace=True)
        # left_merge_players_df.rename(columns={'index': 'match_teams'}, inplace=True)

        # rename match_teams_x to match_teams and season_match_teams_x to season_match_teams
        left_merge_players_df.rename(columns={'season_match_teams_x': 'season_match_teams', 'match_teams_x': 'match_teams'}, inplace=True)

        # log the start of the clean_duplicate_columns function
        clean_duplicate_columns_start_time = log_start_of_function('clean_duplicate_columns')

        left_merge_players_df = clean_duplicate_columns(left_merge_players_df)

        log_end_of_function('clean_duplicate_columns', clean_duplicate_columns_start_time, app_start_time)

        # Cut the dataframes into smaller ones
        # print fstring
        print(f"Chunking the dataframes...")

        # log the start of the cut_df function
        cut_df_start_time = log_start_of_function('cut_df')

        left_merge_players_dict = cut_df(left_merge_players_df, ['match_teams', 'season_match_teams'])
        only_results_dict = cut_df(only_results_df, ['match_teams', 'season_match_teams'])

        # log dict details
        log_dict_contents(left_merge_players_dict, 'left_merge_players_dict')
        log_dict_contents(only_results_dict, 'only_results_dict')

        print(f"Chunking complete.\n--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start ---")

        # log the end of the cut_df function
        log_end_of_function('cut_df', cut_df_start_time, app_start_time)

        # Calculate per90 stats for each dataframe in the players_df_dict
        print(f"Calculating per90 stats...")
        # log the start of the calculate_per90s function
        calculate_per90s_start_time = log_start_of_function('calculate_per90s')

        left_merge_players_dict = calculate_per90s(left_merge_players_dict)

        print(f"Per90 stats calculated.\n--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start ---")

        # log the end of the calculate_per90s function
        log_end_of_function('calculate_per90s', calculate_per90s_start_time, app_start_time)

        # Save the dataframes
        print(f"Saving the Dictionaries and Dataframes as CSVs...")

        # log the start of the save_as_csvs function
        save_csvs_start_time = log_start_of_function('save_csvs')

        # Saving left_merge_players_df, left_merge_players_dict, only_results_df, and only_results_dict
        save_csvs(left_merge_players_df, 'left_merge_players_df')
        save_csvs(only_results_df, 'only_results_df')

        # log end of save_csvs function
        log_end_of_function('save_csvs', save_csvs_start_time, app_start_time)

        # log the start of the save_dicts function
        save_dicts_start_time = log_start_of_function('save_dicts')

        save_dicts(left_merge_players_dict, 'left_merge_players_dict')
        save_dicts(only_results_dict, 'only_results_dict')

        # log end of save_dicts function
        log_end_of_function('save_dicts', save_dicts_start_time, app_start_time)

        print(f"Dictionaries saved as CSVs.\n--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start ---")

        # log the end of the save_as_csvs function
        print(f"Saving the Dataframes as Databases...")

        # log the start of the save_as_dbs function
        save_as_dbs_start_time = log_start_of_function('save_as_dbs')

        save_as_dbs(left_merge_players_df, only_results_df, FINAL_DB_PATH)

        print(f"Dataframes saved as Databases.\n--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) ---")

        # log the end of the save_as_dbs function
        log_end_of_function('save_as_dbs', save_as_dbs_start_time, app_start_time)

        print(f"Dictionaries saved as CSVs & Dataframes saved as Databases saved.\n--- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start ---")
        
    finally:
        conn.close()

    # Print the execution time
    print(f"Script executed in {time.time() - start_time} seconds.")

    # log end of main() function
    log_end_of_function('main()', main_function_start_time, app_start_time)

    return left_merge_players_df, only_results_df

if __name__ == "__main__":
    left_merge_players_df, only_results_df = main()


