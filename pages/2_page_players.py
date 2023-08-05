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

sys.path.append(os.path.abspath(os.path.join('./scripts')))

from constants import color1, color2, color3, color4, color5, cm

from constants 

from functions import * 

# load player data
players_data, _ = load_data_from_csv()

# clean player data
players_data, _ = clean_data(players_data, _)

# calculate per90 player data
df_per90 = calculate_per90(players_data)
df_per90 = df_per90.reset_index(drop=True)

# rename columns
players_only_df = rename_columns(df_per90)

# print out unqiue seasons  
print(f"Unique Seasons: {players_only_df['season'].unique()}")

# print columns to list
cols = players_only_df.columns.tolist()
print(cols)

# process player data
process_player_datav2(players_only_df)

# reorder the columns with player then season then rest of cols
first_cols = ['Player', 'Season']
rest_of_cols = [col for col in players_only_df.columns if col not in first_cols]
players_only_df = players_only_df[first_cols + rest_of_cols]

# selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = dropdown_for_player_stats(players_only_df)

# def main():
#     # Set the title of the app
#     st.title('Premier League Player Stats')

#     # load the data
#     players_data, _ = load_data_from_csv()

#     # clean the data
#     players_data, _ = clean_data(players_data, _)

#     # per90 player data
#     df_per90 = calculate_per90(players_data)

#     # rename columns
#     players_only_df = rename_columns(df_per90)

#     # process player data
#     players_data, season_dfs, teams_dfs, vs_teams_dfs, ages_dfs, nations_dfs, positions_dfs, referees_dfs, venues_dfs = process_player_datav2(players_data)

#     # reorder the columns with player then season then rest of cols
#     first_cols = ['Player', 'Season']
#     rest_of_cols = [col for col in players_only_df.columns if col not in first_cols]
#     players_only_df = players_only_df[first_cols + rest_of_cols]

#     # selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = dropdown_for_player_stats(players_only_df)
    