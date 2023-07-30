# Import statements and other dependencies
import pandas as pd
import numpy as np
import uuid
import glob
import os
import sqlite3

# Define the output and data load paths
output_path = '/Users/hogan/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/streamlit/data'
data_to_load_path = output_path

def load_data():
    list_of_csvs = ['all_seasons_combined_df', 'df_2017_2018', 'df_2018_2019', 'df_2019_2020', 'df_2020_2021', 'df_2021_2022', 'df_2022_2023', 'df_1992_2016']
    list_of_files = [data_to_load_path + '/' + csv + '.csv' for csv in list_of_csvs]

    dict_of_dfs = {csv: pd.read_csv(file) for csv, file in zip(list_of_csvs, list_of_files)}
    return dict_of_dfs

def clean_results(results_df):
    drop_cols = ['score', 'match_report', 'notes']
    results_df = results_df.drop(columns=[col for col in drop_cols if col in results_df])

    desired_columns_order = ['season', 'gameweek', 'home_team', 'home_xg', 'away_xg', 'away_team', 'home_score', 'away_score', 'date', 'referee']
    results_df = results_df.loc[:, desired_columns_order]

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
    players_df['match_teams'] = players_df.apply(lambda row: '_'.join(sorted([row['team'], row['opponent']])), axis=1)
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

def save_dfs(df_dict, csv_path='data/chunked_data/', db_path='data/chunked_data/sqlite3_db_files/'):
    """
    Summary: 
        saves each df in df_dict as a csv file and as a SQLite3 db file

    Args:
        df_dict (dict): dictionary of dataframes
        csv_path (str): directory to save csv files
        db_path (str): directory to save SQLite3 db files

    Returns:
        None
    """
    # Ensure directories exist
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(db_path, exist_ok=True)
    
    for key, df in df_dict.items():
        # Convert key to a string if it's not already
        if not isinstance(key, str):
            key = str(key)

        # Save as a csv file
        csv_file_path = os.path.join(csv_path, f"{key}.csv")
        df.to_csv(csv_file_path, index=False)

        # Save as a SQLite3 db file
        db_file_path = os.path.join(db_path, f"{key}.db")
        conn = sqlite3.connect(db_file_path)
        df.to_sql(key, conn, if_exists='replace', index=False)
        conn.close()

def main():
    # Call load_data() function
    dict_of_dfs = load_data()

    # players_df is just the all_seasons_combined_df
    players_df = dict_of_dfs['all_seasons_combined_df']

    # remove the all_seasons_combined_df from the dict_of_dfs
    dict_of_dfs.pop('all_seasons_combined_df')

    # Concatenate the dataframes in the dictionary vertically
    results_df = pd.concat(dict_of_dfs.values(), axis=0)

    # Clean the dataframes
    results_df = clean_results(results_df)
    players_df = clean_players(players_df)

    # Cut the dataframes into smaller ones
    results_df_dict = cut_df(results_df, ['match_teams', 'season_match_teams'])
    players_df_dict = cut_df(players_df, ['match_teams', 'season_match_teams'])

    # Calculate per90 stats for each dataframe in the players_df_dict
    players_df_dict = calculate_per90s(players_df_dict)

    # Save the dataframes
    save_dfs(results_df_dict, 'data/chunked_data/results/', 'data/chunked_data/sqlite3_db_files/results/')
    save_dfs(players_df_dict, 'data/chunked_data/players/', 'data/chunked_data/sqlite3_db_files/players/')

    # merge the players_df and results_df on the matchup_merge_key
    players_results_df = pd.merge(players_df, results_df, on='matchup_merge_key', how='inner')

if __name__ == "__main__":
    main()
