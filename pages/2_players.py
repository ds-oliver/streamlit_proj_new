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
def app(raw_data):
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

    bundesliga_df = big5_players_data[big5_players_data['League'] == 'Bundesliga']

    serie_a_df = big5_players_data[big5_players_data['League'] == 'Serie A']

    la_liga_df = big5_players_data[big5_players_data['League'] == 'La Liga']

    ligue_1_df = big5_players_data[big5_players_data['League'] == 'Ligue 1']


    # get per90 stats for each player by dividing each numerical column by the 90s column
    for df in [premier_league_df, bundesliga_df, serie_a_df, la_liga_df, ligue_1_df]:
        # set index to 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Matches', 'Position Category', 'League'
        # drop 'Matches Played', 'Rk', 'Born', 'Age', 'Games Played', 'Minutes Played'
        df_cols = ['Matches Played', 'Rk', 'Born', 'Age', 'Games Played', 'Minutes Played']
        for col in df_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
            else:
                continue

        df.set_index(['Player', 'Nation', 'Pos', 'Team', 'Matches', 'Position Category', 'League', 'Season'], inplace=True)
        # divide each column by the 90s column
        df_cols = df.columns.tolist()

        for col in df_cols:
            # avoid calculating per90s on 90s column and any with % or percent in the name
            if col != '90s' and '%' not in col and 'percent' not in col and 'per90' not in col and 'Per90' not in col and 'Per 90' not in col and 'per 90' not in col and 'Minutes' not in col:
                df[f"{col} Per90"] = df[col] / df['90s']
                # normalize the per90 columns by dividing by the max value and handling 0s
                df_norm = df.copy()

                df_norm[f"{col} Per90"] = df[f"{col} Per90"].apply(lambda x: x / df[f"{col} Per90"].max() if x != 0 else 0)

    # reset index for each df and rename columns, drop 90s column
    for df in [premier_league_df, bundesliga_df, serie_a_df, la_liga_df, ligue_1_df]:
        df.reset_index(inplace=True)
        df = df.drop('90s', axis=1)

    # check columns that have a range between
    # rename columns using the rename_columns function
    premier_league_df = rename_columns(premier_league_df)
    bundesliga_df = rename_columns(bundesliga_df)
    serie_a_df = rename_columns(serie_a_df)
    la_liga_df = rename_columns(la_liga_df)
    ligue_1_df = rename_columns(ligue_1_df)

    print(premier_league_df[premier_league_df['Season'] == '2023'].head(10))

    pl_2023 = premier_league_df[premier_league_df['Season'] == '2023']

    print(pl_2023['Assists'].describe())

    # print(premier_league_df.columns)

    print(premier_league_df.columns)
    
    return premier_league_df, bundesliga_df, serie_a_df, la_liga_df, ligue_1_df

premier_league_df, bundesliga_df, serie_a_df, la_liga_df, ligue_1_df = app(big5_players_csv)

premier_league_df = process_player_data(premier_league_df)


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
    