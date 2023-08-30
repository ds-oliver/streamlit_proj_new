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

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

print("Scripts path:", scripts_path)

print(sys.path)

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, matches_col_groups

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, matches_data, shots_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, create_custom_cmap, style_dataframe_custom, style_tp_dataframe_custom, load_csv, add_construction, round_and_format

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
    players_df = load_csv(matches_data)

    # capitalize columns
    shots_df.columns = [col.capitalize() for col in shots_df.columns]
    players_df.columns = [col.capitalize() for col in players_df.columns]

    print(shots_df.head())
    print(players_df.head())

    # filter for only data where outcome does not equal 'Off Target'
    shots_df = shots_df[shots_df['Outcome'] != 'Off Target']

    # team_stats = players_df.groupby(['Team', 'Gameweek'], as_index=False).sum()

    # team_stats.reset_index(drop=True, inplace=True)

    # print(team_stats.head())

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

    # filter for selected match in players_df where players_df['team'] and players_df['opponent'] are in shots_df['matchup']
    players_df = players_df[(players_df['Team'].isin([select_match.split(' vs. ')[0], select_match.split(' vs. ')[1]])) & (players_df['Opponent'].isin([select_match.split(' vs. ')[0], select_match.split(' vs. ')[1]]))]

    # print unique values of players_df['team'] and players_df['opponent']
    print(players_df['Team'].unique())
    print(players_df['Opponent'].unique())

    matches_col_groups = {key.capitalize(): [col.capitalize() for col in value] for key, value in matches_col_groups.items()}

    # select from matches_cols_groups
    select_col_group = st.selectbox('Select Stats Category', matches_col_groups, key='matches_col_groups')

    selected_columns = matches_col_groups[select_col_group]

    print(f"selected col group: {selected_columns}")

    # if column name contains '_pct' add column to list of columns to show grouped by mean
    cols_to_show_as_mean = []
    cols_to_show_as_sum = []
    if selected_columns:
        print(f"select_col_group is not empty: {selected_columns}")
        for col in selected_columns:
            if '_pct' in col:
                print('pct in select col group')
                print(f"col: {col}") 
                cols_to_show_as_mean.append(col) 
            else:
                cols_to_show_as_sum.append(col)
    else:
        print(f"select_col_group is empty: {selected_columns}")

    print(f"cols to show as mean: {cols_to_show_as_mean}")
    print(f"cols to show as sum: {cols_to_show_as_sum}")

    aggregation_functions = {col: 'sum' if col in cols_to_show_as_sum else 'mean' for col in cols_to_show_as_mean}
    # Create an aggregation function dictionary based on the above lists
    aggregation_functions = {}
    for col in selected_columns:
        if col in cols_to_show_as_mean:
            aggregation_functions[col] = 'mean'
        elif col in cols_to_show_as_sum:
            aggregation_functions[col] = 'sum'

    # Use the above aggregation functions dictionary to aggregate your DataFrame
    # Example assuming 'Team' and 'Gameweek' are the grouping columns
    team_df = players_df.groupby(['Team', 'Gameweek']).agg(aggregation_functions).reset_index()

    # apply round_and_format to team_df
    team_df = team_df.applymap(round_and_format)

    columns_to_show_players = ['Player'] + [col for col in selected_columns if col in players_df.columns]

    columns_to_show_team = ['Team'] + [col for col in selected_columns if col in team_df.columns]
    # divider
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        home_team = select_match.split(' vs. ')[0]
        st.subheader(f'{home_team}')
        st.write('*')
        # filter home_team_stats for home_team
        home_team_stats = team_df[team_df['Team'] == home_team]
        
        # get players data from where players_df['team'] == home_team
        home_players_stats = players_df[players_df['Team'] == home_team]
        # reset index
        home_players_stats.reset_index(drop=True, inplace=True)

        print(home_players_stats.head())

        styled_home_players_stats_df = style_tp_dataframe_custom(home_players_stats[columns_to_show_players], columns_to_show_players, False)

        styled_home_team_df = style_tp_dataframe_custom(home_team_stats[columns_to_show_team], columns_to_show_team, False)

        print(f"players columns: {columns_to_show_players}")
        print(f"team columns: {columns_to_show_team}")

        st.dataframe(home_team_stats[columns_to_show_team].style.apply(lambda _: styled_home_team_df, axis=None), height=(len(home_team_stats) * 35), width=1200)

        st.dataframe(home_players_stats[columns_to_show_players].style.apply(lambda _: styled_home_players_stats_df, axis=None), height=(len(home_players_stats) * 35), width=1200)

    with col2:
        away_team = select_match.split(' vs. ')[1]
        st.subheader(f'{away_team}')
        st.write('*')
        # get players data from where players_df['team'] == away_team
        away_players_stats = players_df[players_df['Team'] == away_team]
        # reset index
        away_players_stats.reset_index(drop=True, inplace=True)

        away_teams_stats = team_df[team_df['Team'] == away_team]

        # reset index
        away_teams_stats.reset_index(drop=True, inplace=True)

        print(away_teams_stats.head())

        print(away_players_stats.head())

        styled_home_players_df = style_tp_dataframe_custom(away_players_stats[columns_to_show_players], columns_to_show_players, False)

        styled_away_team_df = style_tp_dataframe_custom(away_teams_stats[columns_to_show_team], columns_to_show_team, False)

        print(columns_to_show_players)

        # # set players column as index
        # away_players.set_index('Player', inplace=True)

        st.dataframe(away_teams_stats[columns_to_show_team].style.apply(lambda _: styled_away_team_df, axis=None), height=(len(away_teams_stats) * 35), width=1200)

        st.dataframe(away_players_stats[columns_to_show_players].style.apply(lambda _: styled_home_players_df, axis=None), height=(len(away_players_stats) * 35), width=1200)

        

if __name__ == "__main__":
    main()