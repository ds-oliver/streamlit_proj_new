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

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_col_groups, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols

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

def process_matches_data(matches_data, temp_data, matches_drop_cols):
    matches_df = read_data(matches_data)
    temp_df = read_data(temp_data)
    print("Shape of matches_df before merging:", matches_df.shape)

    matches_df = pd.merge(matches_df, temp_df[['Player', 'Position', 'Team']], left_on='player', right_on='Player', how='left')
    print("Shape of matches_df after merging:", matches_df.shape)

    # drop the 'Player' and 'team' columns
    matches_df.drop(columns=['Player', 'team'], inplace=True)

    matches_df.rename(columns={'Position': 'Pos'}, inplace=True)
    matches_df.rename(columns={'Team': 'team'}, inplace=True)

    # rename GW column to 'gw'
    matches_df.rename(columns={'gameweek': 'GW'}, inplace=True)
    
    for col in matches_drop_cols:
        if col in matches_df.columns:
            matches_df.drop(columns=col, inplace=True)

    # Apply the replace method only if the value is a string
    matches_df['team'] = matches_df['team'].apply(lambda x: x.replace(' Player Stats', '') if isinstance(x, str) else x)

    matches_df.drop_duplicates(subset=['player', 'GW'], inplace=True)

    print("Columns in matches_df after processing:", matches_df.columns.tolist())

    # capitalize format for the columns
    matches_df.columns = [col.capitalize() for col in matches_df.columns.tolist()]

    print("Columns in matches_df after capitalizing:", matches_df.columns.tolist())

    MATCHES_DEFAULT_COLS = matches_default_cols

    # capitalize format for the columns
    MATCHES_DEFAULT_COLS = [col.capitalize() for col in MATCHES_DEFAULT_COLS]

    print("Default columns:", MATCHES_DEFAULT_COLS)
    
    return matches_df, MATCHES_DEFAULT_COLS


@st.cache_resource
def load_shots_data(shots_data):
    return read_data(shots_data)

def create_top_performers_table(matches_df, selected_group, selected_columns):
    # write header that says 'Top Performers in {GW}"
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
    matches_df, MATCHES_DEFAULT_COLS = process_matches_data(matches_data, temp_default, matches_drop_cols)
    
    shots_df = load_shots_data(shots_data)
    date_of_update = datetime.fromtimestamp(os.path.getmtime(matches_data)).strftime('%d %B %Y')
    return matches_df, shots_df, date_of_update, MATCHES_DEFAULT_COLS

def display_date_of_update(date_of_update):
    st.sidebar.write(f'Last data refresh: {date_of_update}')

def main():
    # Load the data
    matches_df, shots_df, date_of_update, MATCHES_DEFAULT_COLS = process_data()

    print(matches_df.columns.tolist())

    display_date_of_update(date_of_update)

    # state at the top of the page as header the grouping option selected
    st.header(f"Premier League Individual Players' Data:")

    # create radio button for 'Starting XI' or 'All Featured Players'
    featured_players = st.sidebar.radio("Select Featured Players", ('Starting XI', '> 55 Minutes Played', 'All Featured Players'))

    # Filter by specific player
    player_data = matches_df[matches_df['Player'] == 'Marcus Rashford']

    # Check the 'started' column for that player across GWs
    print(player_data[['GW', 'Started']])

    # # print count of featured players
    # st.sidebar.write(f'Number of featured players: {matches_df.shape[0]}')

    # # print count of starting XI
    # st.sidebar.write(f'Number of starting XI: {matches_df[matches_df["started"] > 0].shape[0]}')

    st.divider()  # 👈 Draws a horizontal rule

    # filter the dataframe based on the radio button selected
    if featured_players == '> 55 Minutes Played':
        matches_df = matches_df[matches_df['Minutes'] > 55]
    elif featured_players == 'Starting XI':
        matches_df = matches_df[matches_df['Started'] > 0]

    GW_range = st.sidebar.slider('GW range', min_value=matches_df['GW'].min(), max_value=matches_df['GW'].max(), value=(matches_df['GW'].min(), matches_df['GW'].max()), step=1)

    GW_range = list(GW_range)

    matches_df = matches_df[(matches_df['GW'] >= GW_range[0]) & (matches_df['GW'] <= GW_range[1])]

    print("Shape of matches_df after filtering by featured players:", matches_df.shape)

    matches_df = matches_df[(matches_df['GW'] >= GW_range[0]) & (matches_df['GW'] <= GW_range[1])]
    print("Shape of matches_df after filtering by GW range:", matches_df.shape)


    # if GW_range list has more than 1 element, group by MATCHES_DEFAULT_COLS
    if GW_range[0] != GW_range[1]:
        st.info(f'Grouping data from **:red[GW {GW_range[0]}]** to **:red[GW {GW_range[1]}]**', icon='ℹ')

        # Define aggregation functions for numeric and non-numeric columns
        aggregation_functions = {col: 'sum' if matches_df[col].dtype in [np.float64, np.int64] else 'first' for col in matches_df.columns}
        aggregation_functions['Player'] = 'first'
        aggregation_functions['Team'] = 'first'
        aggregation_functions['Pos'] = 'first' # Aggregating by the first occurrence of position
        aggregation_functions['GW'] = 'nunique' # Counting the number of GWs
        aggregation_functions['Started'] = 'sum' # Summing the number of starts

        # Group by player, team, and position, and apply the aggregation functions
        matches_df = matches_df.groupby(['Player', 'Team', 'Pos'], as_index=False).agg(aggregation_functions)
        print("Shape of matches_df after grouping by player, team, and position:", matches_df.shape)

        # Rename the 'GW' column to 'games played'
        matches_df.rename(columns={'GW': 'GP'}, inplace=True)
        # rename the 'started' column to 'games_starts'
        matches_df.rename(columns={'Started': 'GS'}, inplace=True)
        

        # Update MATCHES_DEFAULT_COLS
        MATCHES_DEFAULT_COLS = [col if col != 'GW' else 'GP' for col in MATCHES_DEFAULT_COLS]
        MATCHES_DEFAULT_COLS = [col if col != 'started' else 'GS' for col in MATCHES_DEFAULT_COLS]

    else:
        # show st.info() message of the GW selected
        st.info(f'GW {GW_range[0]} selected')

    st.info(f'**:green[{matches_df.shape[0]}]** players found within the parameters selected', icon='ℹ')

    st.divider()  # 👈 Draws a horizontal rule

    # User selects the group and columns to show
    selected_group = st.sidebar.selectbox("Select Stats Grouping", list(matches_col_groups.keys()))
    selected_columns = matches_col_groups[selected_group]
    columns_to_show = MATCHES_DEFAULT_COLS + [col for col in selected_columns if col in matches_df.columns]

    # Styling DataFrame
    styled_df = style_dataframe(matches_df[columns_to_show], selected_columns=selected_columns)

    # display the dataframe
    st.dataframe(matches_df[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=50 * 20)


if __name__ == "__main__":
    main()