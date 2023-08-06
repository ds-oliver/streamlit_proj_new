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

from files import big5_players_csv, players_matches_csv, teams_matches_csv

from functions import load_data_from_csv, rename_columns, clean_data, create_multiselect_seasons, create_dropdown_teams, filter_df_by_team_and_opponent, display_qual_stats, display_quant_stats, get_results_df, match_quick_facts, get_teams_stats, show_teams_stats_v2

# suppress settings

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_resource
def app_processing(raw_data):
    # load player data from csv
    players_matches_data, teams_matches_data = load_data_from_csv()

    # clean data
    players_matches_data, teams_matches_data = clean_data(players_matches_csv, teams_matches_csv)

    # print head of df, columns, and shape
    print(players_matches_csv.head())

    # print(big5_players_data.columns)
    print(players_matches_csv.shape)

    return players_matches_data, teams_matches_data

def app_selections(players_matches_data, teams_matches_data):

    # call create_multiselect_seasons
    _, filtered_data = create_multiselect_seasons(players_matches_data)

    selected_team, selected_opponent, filtered_data = create_dropdown_teams(filtered_data)

    selected_teams_df, _ = filter_df_by_team_and_opponent(filtered_data, selected_team, selected_opponent)

    display_qual_stats(selected_teams_df, selected_team, selected_opponent)

    display_quant_stats(selected_teams_df, selected_team, selected_opponent)

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

    # create multiselect for seasons
    seasons = create_multiselect_seasons(big5_players_data)

    # create multiselect for teams
    teams = create_dropdown_teams(big5_players_data)

    # create multiselect for opponents
    opponents = create_dropdown_teams(big5_players_data)

    # create multiselect for player names
    player_names = create_dropdown_teams(big5_players_data)

    # filter df by team and opponent
    df = filter_df_by_team_and_opponent(big5_players_data, teams, opponents)

    # display qual stats
    display_qual_stats(df)

    # display quant stats
    display_quant_stats(df)

    # get results df
    results_df = get_results_df(df)

    # match quick facts
    match_quick_facts(results_df)

    # get teams stats
    teams_stats = get_teams_stats(results_df)

    # show