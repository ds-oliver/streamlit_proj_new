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
sys.path.append(os.path.abspath(os.path.join('./constants')))
sys.path.append(os.path.abspath(os.path.join('./files')))
sys.path.append(os.path.abspath(os.path.join('./functions')))


from constants import color1, color2, color3, color4, color5, cm

from files import big5_players_csv, players_matches_csv, teams_matches_csv

from functions import load_data_from_csv, rename_columns, clean_data, create_multiselect_seasons, create_dropdown_teams, filter_df_by_team_and_opponent, display_qual_stats, display_quant_stats, get_results_df, match_quick_facts, get_teams_stats, show_teams_stats_v2

# suppress settings

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

def app_processing(raw_data):
    # load player data from csv
    players_df, results_df = load_data_from_csv() # Renamed variables

    # clean data
    players_matches_data, teams_matches_data = clean_data(players_df, results_df) # Fixed call

    # print head of df, columns, and shape
    print(players_matches_data.head())

    print(players_matches_data.shape)

    return players_matches_data, teams_matches_data


def app_selections(players_matches_data, teams_matches_data):

    # call create_multiselect_seasons
    _, filtered_data = create_multiselect_seasons(players_matches_data)

    selected_team, selected_opponent, filtered_data = create_dropdown_teams(filtered_data)

    selected_teams_df, _ = filter_df_by_team_and_opponent(filtered_data, selected_team, selected_opponent)

    return selected_teams_df, selected_team, selected_opponent

def app_display(selected_teams_df, selected_team, selected_opponent):
    
    display_qual_stats(selected_teams_df, selected_team, selected_opponent)

    display_quant_stats(selected_teams_df, selected_team, selected_opponent)

    results_df = get_results_df(selected_teams_df, selected_team, selected_opponent)


def app():
    
    # title
    st.title('Matchups')

    # sidebar
    st.sidebar.header('Matchups')

    # load data
    players_matches_data, teams_matches_data = app_processing(players_matches_csv)

    # selections
    selected_teams_df, selected_team, selected_opponent = app_selections(players_matches_data, teams_matches_data)

    # display
    app_display(selected_teams_df, selected_team, selected_opponent)

if __name__ == '__main__':
    app()