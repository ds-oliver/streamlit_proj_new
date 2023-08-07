import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging
import sqlite3
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plost
import plotly.express as px
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import unicodedata
import plotly.graph_objects as go


warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join('./scripts')))

from constants import color1, color2, color3, color4, color5, cm

from files import big5_players_csv

from functions import dropdown_for_player_stats, rename_columns, process_player_data, convert_to_int, min_max_scale

# suppress settings

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_resource
def app_process(raw_data):
    # load player data from csv
    big5_players_data = pd.read_csv(big5_players_csv)

    big5_players_data.fillna(0, inplace=True)

    # print head of df, columns, and shape
    print(big5_players_data.head())
    # print(big5_players_data.columns)
    print(big5_players_data.shape)

    # print out categorical columns
    # print(big5_players_data.select_dtypes(include=['object']).columns)

    # drop Comp column
    big5_players_data.drop('Comp', axis=1, inplace=True)

    # strip whitespace and unidecode the player names, league values and team values
    big5_players_data['Player'] = big5_players_data['Player'].str.strip()
    big5_players_data['Player'] = big5_players_data['Player'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    big5_players_data['League'] = big5_players_data['League'].str.strip()
    big5_players_data['League'] = big5_players_data['League'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    big5_players_data['Squad'] = big5_players_data['Squad'].str.strip()
    big5_players_data['Squad'] = big5_players_data['Squad'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # rename squad to team
    big5_players_data = big5_players_data.rename(columns={'Squad': 'Team'})

    # create a df for each league
    premier_league_df = big5_players_data[big5_players_data['League'] == 'Premier League']

    return premier_league_df

    # bundesliga_df = big5_players_data[big5_players_data['League'] == 'Bundesliga']

    # serie_a_df = big5_players_data[big5_players_data['League'] == 'Serie A']

    # la_liga_df = big5_players_data[big5_players_data['League'] == 'La Liga']

    # ligue_1_df = big5_players_data[big5_players_data['League'] == 'Ligue 1']

def normalize_and_clean_data(df):
    
    print(f"Running normalize_and_clean_data function...")
    # Drop unnecessary columns
    print(f"Columns before dropping unnecessary columns: {df.columns.tolist()}")
    df = df.drop(['Rk', 'matches_played', 'games_started', 'Born', 'Age', 'Games Played', 'minutes_played', 'Matches', 'League'], axis=1, errors='ignore')
    
    # Set new index
    idx_cols = ['Player', 'Nation', 'Pos', 'Team', 'Position Category', 'Season']
    if all(col in df.columns for col in idx_cols):
        df.set_index(idx_cols, inplace=True)

    print(f"Columns after dropping unnecessary columns and setting index: {df.columns.tolist()}")

    # print out columns grouped by data type and count
    print(df.columns.to_series().groupby(df.dtypes).count())
    print(f"Total columns: {len(df.columns)}")

    print(f"Normalizing cols...")
    list_of_cols_passed = []
    cols_to_convert_to_per90s = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and '90s' not in col and '%' not in col and 'percent' not in col and 'per90' not in col and 'Per90' not in col and 'Per 90' not in col and 'per 90' not in col and 'Minutes' not in col]

    cols_to_skip = [col for col in df.columns if col not in cols_to_convert_to_per90s]
    print(f"Cols to skip: {cols_to_skip}")
    print(f"Cols to pass: {cols_to_convert_to_per90s}")
    # Get the columns to convert
    # Loop through the columns and convert them
    for col in cols_to_convert_to_per90s:
        df[f"{col} Per90"] = df.apply(lambda row: row[col] / row['90s'] if row['90s'] != 0 else 0, axis=1)
        max_val = df[f"{col} Per90"].max()
        df[f"{col} Per90"] = df[f"{col} Per90"].apply(lambda x: x / abs(max_val) if x != 0 else 0)
        # drop the original col, then rename the new col to the original col name
        df = df.drop(col, axis=1, errors='ignore')
        df = df.rename(columns={f"{col} Per90": col})

        
        # append the new col name to list of cols passed
        list_of_cols_passed.append(col)

    print(f"Normalizing complete...")

    df = df.drop('90s', axis=1, errors='ignore')

    print(f"List of cols passed: {list_of_cols_passed}")
    print(f"Cols passed minus cols skipped: {len(list_of_cols_passed) - len(cols_to_skip)}")
    #print total new cols
    print(f"Total new cols: {len(df.columns)}")

    # reset index
    df = df.reset_index()

    # new df only keeping idx_cols and list_of_cols_passed 
    df = df[idx_cols + list_of_cols_passed]

    # Assuming rename_columns function exists and is valid
    df = rename_columns(df)

    print(f"Columns after normalizing and cleaning: {df.columns.tolist()}")

    # check that columns are between 0 and 1
    if df[col].min() >= 0 and df[col].max() <= 1:
        print(f"{col} is between 0 and 1")
    
    # print cols that have negative values
    # check that columns are between 0 and 1, take the absolute value if negative
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].min() < 0:
            df[col] = df[col].apply(lambda x: abs(x))
        elif df[col].dtype in ['int64', 'float64'] and df[col].min() >= 0 and df[col].max() <= 1:
            print(f"{col} is between 0 and 1")

    return df

premier_league_df = app_process(big5_players_csv)

premier_league_df = normalize_and_clean_data(premier_league_df)

player_df = process_player_data(premier_league_df)

# turn the values into percentiles
# season_dfs = []  # list to collect all season DataFrames
# for season in seasons:
#     # get the df for the season
#     season_df = premier_league_df_scaled[premier_league_df_scaled['Season'] == season].copy()  # create a copy to avoid SettingWithCopyWarning
#     # get the columns
#     cols = season_df.columns.tolist()
#     # remove the columns that are not numerical
#     cols.remove('Player')
#     cols.remove('Nation')
#     cols.remove('Pos')
#     cols.remove('Team')
#     cols.remove('Matches')
#     cols.remove('Position Category')
#     cols.remove('League')
#     cols.remove('Season')
#     # loop through the columns and turn them into percentiles
#     for col in cols:
#         # get the percentile values
#         season_df[f"{col} Percentile"] = season_df[col].rank(pct=True)
#     # reset the index
#     season_df.reset_index(drop=True, inplace=True)
#     season_df.fillna(0, inplace=True)
#     # collect the modified season DataFrame
#     season_dfs.append(season_df)
# # create the final DataFrame by concatenating all season DataFrames
# premier_league_df_scaled = pd.concat(season_dfs, axis=0).reset_index(drop=True)

# # process_player_data function
            
# create a new df 

# # load player data
# players_data, _ = load_data_from_csv()

# # clean player data
# players_data, _ = clean_data(players_data, _)

# # calculate per90 player data
# df_per90 = calculate_per90(players_data)
# df_per90 = df_per90.reset_index(drop=True)

# # rename columns
# players_only_df = rename_columns(df_per90)

# # print out unqiue seasons  
# print(f"Unique Seasons: {players_only_df['season'].unique()}")

# # print columns to list
# cols = players_only_df.columns.tolist()
# print(cols)

# # process player data
# process_player_datav2(players_only_df)

# # reorder the columns with player then season then rest of cols
# first_cols = ['Player', 'Season']
# rest_of_cols = [col for col in players_only_df.columns if col not in first_cols]
# players_only_df = players_only_df[first_cols + rest_of_cols]

# # selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = dropdown_for_player_stats(players_only_df)

# # def main():
# #     # Set the title of the app
# #     st.title('Premier League Player Stats')

# #     # load the data
# #     players_data, _ = load_data_from_csv()

# #     # clean the data
# #     players_data, _ = clean_data(players_data, _)

# #     # per90 player data
# #     df_per90 = calculate_per90(players_data)

# #     # rename columns
# #     players_only_df = rename_columns(df_per90)

# #     # process player data
# #     players_data, season_dfs, teams_dfs, vs_teams_dfs, ages_dfs, nations_dfs, positions_dfs, referees_dfs, venues_dfs = process_player_datav2(players_data)

# #     # reorder the columns with player then season then rest of cols
# #     first_cols = ['Player', 'Season']
# #     rest_of_cols = [col for col in players_only_df.columns if col not in first_cols]
# #     players_only_df = players_only_df[first_cols + rest_of_cols]

# #     # selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = dropdown_for_player_stats(players_only_df)
    