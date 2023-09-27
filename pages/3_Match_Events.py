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
from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, matches_col_groups

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, matches_data, shots_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, create_custom_cmap, style_dataframe_custom, style_tp_dataframe_custom, load_csv, add_construction, round_and_format, load_css

# logger = st.logger

st.set_page_config(
    page_title="Footy Magic",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={

    }
)

js_code = """
    <script>
        setTimeout(function() {
            // Select all elements with the class "glideDataEditor" and "wzg2m5k"
            const elements = document.querySelectorAll(".glideDataEditor wzg2m5k");

            // Loop through all matched elements
            elements.forEach((element) => {
                // Update CSS variables
                element.style.setProperty("--gdg-bg-header", "#370617");
                element.style.setProperty("--gdg-bg-header-has-focus", "#370617");
                element.style.setProperty("--gdg-bg-header-hovered", "#370617");
            });
        }, 1000);  // Execute after a delay of 1000 milliseconds
    </script>
"""

st.markdown(js_code, unsafe_allow_html=True)

load_css()

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

print("Scripts path:", scripts_path)

print(sys.path)

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

    # create a dictionary of player names to positions
    player_to_pos = dict(zip(
        [player for player in load_csv(temp_default)['Player']],
        [pos for pos in load_csv(temp_default)['Position']]
    ))

    # print the number of players in the dictionary
    print(f"Number of players in dictionary: {len(player_to_pos)}")

    shots_df = load_csv(shots_data)

    # print head of dataframe
    print(f"Shots dataframe shape: {shots_df.head()}")

    players_df = load_csv(matches_data)

    # print len of dataframe
    print(f"Players dataframe shape: {len(players_df)}")

    print(f"Players dataframe shape: {players_df.head()}")

    # create position column in shots and players dataframes using player_to_pos dictionary
    players_df['Position'] = np.where(players_df['player'].isin(player_to_pos.keys()), players_df['player'].map(player_to_pos), 'Unknown')

    # Capitalize columns
    shots_df.columns = [col.capitalize() for col in shots_df.columns]
    players_df.columns = [col.capitalize() for col in players_df.columns]

    shots_df = shots_df[shots_df['Outcome'] != 'Off Target']

    if shots_df.empty:
        st.warning("No data to display after filtering.")
        return
    
    select_gw = st.selectbox('Select Gameweek', shots_df['Gameweek'].unique())

    # filter by select_gw
    

    shots_df = shots_df[shots_df['Gameweek'] == select_gw]

    players_df['Team'] = players_df['Team'].apply(lambda x: x.replace(' Player Stats', ''))
    players_df['Opponent'] = players_df['Opponent'].apply(lambda x: x.replace(' Player Stats', ''))
    
    players_df['Matchup'] = players_df['Team'] + ' vs. ' + players_df['Opponent']

    # print unique matchups
    print(f"Unique matchups: {players_df['Matchup'].unique()}")
    
    select_match = st.selectbox('Select Match', players_df['Matchup'].unique())
    
    filter_condition = (players_df['Team'].isin([select_match.split(' vs. ')[0], select_match.split(' vs. ')[1]])) & \
                       (players_df['Opponent'].isin([select_match.split(' vs. ')[0], select_match.split(' vs. ')[1]]))
    players_df = players_df[filter_condition]

    matches_col_groups = {key.capitalize(): [col.capitalize() for col in value] for key, value in matches_col_groups.items()}

    select_col_group = st.selectbox('Select Stats Category', list(matches_col_groups.keys()), key='matches_col_groups')

    selected_columns = matches_col_groups[select_col_group]

    # Aggregation settings
    cols_to_show_as_mean = [col for col in selected_columns if '_pct' in col]
    cols_to_show_as_sum = [col for col in selected_columns if '_pct' not in col]

    aggregation_functions = {col: 'mean' if col in cols_to_show_as_mean else 'sum' for col in selected_columns}
    aggregation_functions['Player'] = 'count'

    # Perform aggregation and reset index
    team_df = players_df.groupby(['Team', 'Gameweek']).agg(aggregation_functions).reset_index()

    # Rename 'Player' column to 'Player Count'
    team_df.rename(columns={'Player': 'Player Count'}, inplace=True)

    # Apply the formatting function
    team_df = team_df.applymap(round_and_format)

    # create Subs column that takes count of players minus 11

    columns_to_show_players = [col for col in selected_columns if col in players_df.columns]
    columns_to_show_team = [col for col in selected_columns if col in team_df.columns]

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        home_team = select_match.split(' vs. ')[0]
        st.subheader(f'{home_team}')
        home_team_stats = team_df[team_df['Team'] == home_team]
        home_team_stats.set_index('Team', inplace=True)

        home_players_stats = players_df[players_df['Team'] == home_team]
        home_players_stats.set_index(['Player', 'Position'], inplace=True)


        styled_home_players_stats_df = style_tp_dataframe_custom(home_players_stats[columns_to_show_players], columns_to_show_players)
        styled_home_team_df = style_tp_dataframe_custom(home_team_stats[columns_to_show_team], columns_to_show_team)

        st.dataframe(home_team_stats[columns_to_show_team].style.apply(lambda _: styled_home_team_df, axis=None))
        st.dataframe(home_players_stats[columns_to_show_players].style.apply(lambda _: styled_home_players_stats_df, axis=None))

    with col2:
        away_team = select_match.split(' vs. ')[1]
        st.subheader(f'{away_team}')
        away_team_stats = team_df[team_df['Team'] == away_team]
        away_team_stats.set_index('Team', inplace=True)

        away_players_stats = players_df[players_df['Team'] == away_team]
        away_players_stats.set_index(['Player', 'Position'], inplace=True)

        styled_away_players_stats_df = style_tp_dataframe_custom(away_players_stats[columns_to_show_players], columns_to_show_players)
        styled_away_team_df = style_tp_dataframe_custom(away_team_stats[columns_to_show_team], columns_to_show_team)

        st.dataframe(away_team_stats[columns_to_show_team].style.apply(lambda _: styled_away_team_df, axis=None))
        st.dataframe(away_players_stats[columns_to_show_players].style.apply(lambda _: styled_away_players_stats_df, axis=None))
    

if __name__ == "__main__":
    main()