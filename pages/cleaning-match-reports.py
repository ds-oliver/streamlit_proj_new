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

# logger = st.logger

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_col_groups, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, other_default_cols

print("Scripts path:", scripts_path)

print(sys.path)

st.set_page_config(
    layout="wide"
)

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, matches_all_data as matches_data, matches_shots_data as shots_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect

def get_color(value, cmap):
    color_fraction = value
    rgba_color = cmap(color_fraction)
    brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
    text_color = 'white' if brightness < 0.7 else 'black'
    return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

def style_dataframe(df, selected_columns):
    cm_coolwarm = cm.get_cmap('inferno')
    object_cmap = cm.get_cmap('gnuplot2')

    # Create an empty DataFrame with the same shape as df
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    for col in df.columns:
        if col == 'player':  # Skip the styling for the 'player' column
            continue
        col_dtype = df[col].dtype  # Get the dtype of the individual column
        if col_dtype in [np.float64, np.int64] and col in selected_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / range_val, cm_coolwarm))
        elif col_dtype == 'object':
            unique_values = df[col].unique().tolist()
            styled_df[col] = df[col].apply(lambda x: get_color(unique_values.index(x) / len(unique_values), object_cmap))
    return styled_df

def read_data(file_path):
    return pd.read_csv(file_path)

def process_matches_data(matches_data, temp_data):
    matches_df = read_data(matches_data)
    temp_df = read_data(temp_data)
    matches_df['fantrax position'] = temp_df['Position']
    # drop matches_drop_cols
    for col in matches_drop_cols:
        if col in matches_df.columns:
            matches_df.drop(columns=col, inplace=True)

    # if ' Player Stats' in 'team' column, then remove ' Player Stats'
    matches_df['team'] = matches_df['team'].apply(lambda x: x.replace(' Player Stats', ''))
    print(matches_df.columns.tolist())
    matches_df.rename(columns={'fantrax position': 'position'}, inplace=True)
    MATCHES_DEFAULT_COLUMNS = ['player', 'gameweek','team', 'position']
    return matches_df, MATCHES_DEFAULT_COLUMNS

def load_shots_data(shots_data):
    return read_data(shots_data)

def create_top_performers_table(matches_df, selected_group, selected_columns):
    # write header that says 'Top Performers in {gameweek}"

    st.write(f'Top Performers in {selected_group}')
    # create a copy of the dataframe
    top_performers_df = matches_df.copy()

    # create a list of the columns to drop
    cols_to_drop = [col for col in top_performers_df.columns if col not in selected_columns]

    # drop the columns
    top_performers_df.drop(columns=cols_to_drop, inplace=True)

    # create a list of the columns to group by
    cols_to_group = [col for col in top_performers_df.columns if col not in ['player', 'team', 'position']]

    # group by the columns
    top_performers_df = top_performers_df.groupby(cols_to_group).mean().reset_index()

    # sort the dataframe by the selected group column
    top_performers_df.sort_values(by=selected_group, ascending=False, inplace=True)



def process_data():
    matches_df, MATCHES_DEFAULT_COLUMNS = process_matches_data(matches_data, temp_default)
    shots_df = load_shots_data(shots_data)
    date_of_update = datetime.fromtimestamp(os.path.getmtime(matches_data)).strftime('%d %B %Y')
    return matches_df, shots_df, date_of_update, MATCHES_DEFAULT_COLUMNS

def display_date_of_update(date_of_update):
    st.sidebar.write(f'Last updated: {date_of_update}')

def main():
    
    # Load the data
    matches_df, shots_df, date_of_update, MATCHES_DEFAULT_COLUMNS = process_data()

    display_date_of_update(date_of_update)

    # Create the sidebar slider to select gameweek a range of gameweek values
    gameweek_range = st.sidebar.slider('Gameweek range', min_value=matches_df['gameweek'].min(), max_value=matches_df['gameweek'].max(), value=(matches_df['gameweek'].min(), matches_df['gameweek'].max()), step=1)

    # create radio button for 'Starting XI' or 'All Featured Players'
    featured_players = st.sidebar.radio("Select Featured Players", ('Starting XI', 'All Featured Players'))

    if featured_players == 'Starting XI':
        matches_df = matches_df[matches_df['started'] == 1]

    # User selects the group and columns to show
    selected_group = st.sidebar.selectbox("Select Stats Grouping", list(matches_col_groups.keys()))

    if selected_group == 'Create Your Own Table':
        selected_stats = st.sidebar.multiselect("Select Columns", list(matches_df.columns), default=other_default_cols)
        columns_to_show = MATCHES_DEFAULT_COLUMNS + selected_stats
        styled_df = style_dataframe(matches_df[columns_to_show], selected_columns=selected_stats)
    else:
        selected_columns = matches_col_groups[selected_group]
        columns_to_show = MATCHES_DEFAULT_COLUMNS + selected_columns
        styled_df = style_dataframe(matches_df[columns_to_show], selected_columns=selected_columns)

    # state at the top of the page as header the grouping option selected
    st.header(f"Premier League Individual Players' Statistics:{selected_group}")
    st.dataframe(matches_df[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=(len(matches_df) * 38) + 50)

    # Create the sidebar multiselect to select the team

    # matches_df

    # shots_df

    # st.dataframe(matches_df, use_container_width=True, height=(len(matches_df) * 38) + 50)
    # st.dataframe(shots_df, use_container_width=True, height=(len(shots_df) * 38) + 50)


if __name__ == "__main__":
    main()