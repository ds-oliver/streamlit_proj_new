import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
# import logging
# import sqlite3
# import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import warnings
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# import unicodedata
import plotly.graph_objects as go
# from bs4 import BeautifulSoup
from matplotlib import cm
from pandas.io.formats.style import Styler
import cProfile
import pstats
import io
import matplotlib.colors as mcolors

# logger = st.logger

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

print("Scripts path:", scripts_path)

print(sys.path)

st.set_page_config(
    layout="wide"
)

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, matches_col_groups

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, matches_data, shots_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, create_custom_cmap, style_dataframe_custom, style_tp_dataframe_custom, load_csv, add_construction

def main():
    
    matches_col_groups = {
    "Standard": matches_standard_cols,
    "Passing": matches_passing_cols,
    "Defense": matches_defense_cols,
    "Possession": matches_possession_cols,
    "Miscellaneous": matches_misc_cols,
    "Passing Types": matches_pass_types
}
    add_construction()
    
    st.title('Match Events')
    st.header('')

    shots_df = load_csv(shots_data)
    match_df = load_csv(matches_data)

    # capitalize columns
    shots_df.columns = [col.capitalize() for col in shots_df.columns]
    match_df.columns = [col.capitalize() for col in match_df.columns]

    print(shots_df.head())
    print(match_df.head())

    # filter for only data where outcome does not equal 'Off Target'
    shots_df = shots_df[shots_df['Outcome'] != 'Off Target']

    # Check if DataFrame is empty
    if shots_df.empty:
        st.warning("No data to display after filtering.")
        return
    
    # provide dropdown for gameweek
    select_gw = st.selectbox('Select Gameweek', shots_df['Gameweek'].unique())

    # filter for selected gameweek
    shots_df = shots_df[shots_df['Gameweek'] == select_gw]
    
    # for each home team away team pair create a new column with 'Home Team vs. Away Team'
    shots_df['Matchup'] = shots_df['Home_team'] + ' vs. ' + shots_df['Away_team']

    # offer selection of matches to show
    select_match = st.selectbox('Select Match', shots_df['Matchup'].unique())

    # filter for selected match in match_df where match_df['team'] and match_df['opponent'] are in shots_df['matchup']
    match_df = match_df[(match_df['Team'].isin([select_match.split(' vs. ')[0], select_match.split(' vs. ')[1]])) & (match_df['Opponent'].isin([select_match.split(' vs. ')[0], select_match.split(' vs. ')[1]]))]
    # print unique values of match_df['team'] and match_df['opponent']
    print(match_df['Team'].unique())
    print(match_df['Opponent'].unique())

    matches_col_groups = {key.capitalize(): [col.capitalize() for col in value] for key, value in matches_col_groups.items()}

    # select from matches_cols_groups
    select_col_group = st.selectbox('Select Stats Category', matches_col_groups, key='matches_col_groups')

    # divider
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        home_team = select_match.split(' vs. ')[0]
        st.subheader(f'{home_team}')
        st.write('*')
        # get players data from where match_df['team'] == home_team
        home_players = match_df[match_df['Team'] == home_team]
        # reset index
        home_players.reset_index(drop=True, inplace=True)

        print(home_players.head())

        selected_columns = matches_col_groups[select_col_group]
    # matches_df[selected_group] = matches_df[selected_group].apply(lambda x: f"{x:.2f}")
        columns_to_show = ['Player'] + [col for col in selected_columns if col in home_players.columns]

        styled_home_df = style_tp_dataframe_custom(home_players[columns_to_show], columns_to_show, False)

        print(columns_to_show)

        st.dataframe(home_players[columns_to_show].style.apply(lambda _: styled_home_df, axis=None), height=(len(home_players) * 35), width=1200)

    with col2:
        away_team = select_match.split(' vs. ')[1]
        st.subheader(f'{away_team}')
        st.write('*')
        # get players data from where match_df['team'] == away_team
        away_players = match_df[match_df['Team'] == away_team]
        # reset index
        away_players.reset_index(drop=True, inplace=True)

        print(away_players.head())

        selected_columns = matches_col_groups[select_col_group]
        # matches_df[selected_group] = matches_df[selected_group].apply(lambda x: f"{x:.2f}")
        columns_to_show = ['Player'] + [col for col in selected_columns if col in away_players.columns]

        styled_home_df = style_tp_dataframe_custom(away_players[columns_to_show], columns_to_show, False)

        print(columns_to_show)

        # # set players column as index
        # away_players.set_index('Player', inplace=True)

        st.dataframe(away_players[columns_to_show].style.apply(lambda _: styled_home_df, axis=None), height=(len(away_players) * 35), width=1200)

        

if __name__ == "__main__":
    main()