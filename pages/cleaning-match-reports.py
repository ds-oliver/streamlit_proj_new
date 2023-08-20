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

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues

print("Scripts path:", scripts_path)

print(sys.path)

st.set_page_config(
    layout="wide"
)

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, matches_all_data, matches_shots_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect

def get_color(value, cmap):
    color_fraction = value
    rgba_color = cmap(color_fraction)
    brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
    text_color = 'white' if brightness < 0.7 else 'black'
    return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

def style_dataframe(df, selected_columns):
    cm_coolwarm = cm.get_cmap('inferno')
    object_cmap = cm.get_cmap('gnuplot2')  # Choose a colormap for object columns

    # Create an empty DataFrame with the same shape as df
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    for col in df.columns:
        if col == 'player':  # Skip the styling for the 'player' column
            continue
        if df[col].dtype in [np.float64, np.int64] and col in selected_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / range_val, cm_coolwarm))
        elif df[col].dtype == 'object':
            unique_values = df[col].unique().tolist()
            styled_df[col] = df[col].apply(lambda x: get_color(unique_values.index(x) / len(unique_values), object_cmap))
    return styled_df

@st.cache_resource
def process_data(matches_all_data, temp_default, matches_shots_data):
    
    df = pd.read_csv(matches_all_data)
    temp_df = pd.read_csv(temp_default)
    df['fantrax position'] = temp_df['Position']

    # drop df['position'] column
    df.drop(columns=['position'], inplace=True)

    # rename 'fantrax position' column to 'position'
    df.rename(columns={'fantrax position': 'position'}, inplace=True)

    # load shots data
    shots_df = pd.read_csv(matches_shots_data)

    # Define default columns
    DEFAULT_COLUMNS = ['player', 'position', 'team', 'games_starts']

    # create timestamp so we can use to display the date of the last data update
    date_of_update = datetime.fromtimestamp(os.path.getmtime(pl_data_gw1)).strftime('%d %B %Y')

    return df, shots_df, DEFAULT_COLUMNS, date_of_update

# we want to add a date of last data update to the page
def display_date_of_update(date_of_update):
    st.sidebar.write(f'Last updated: {date_of_update}')
    
# Function to load the data
@st.cache_resource
def load_data():
    return process_data(pl_data_gw1, temp_default)

def main():
    # Load the data
    matches_data, shots_data, DEFAULT_COLUMNS, date_of_update = load_data()

    st.dataframe(matches_data, use_container_width=True, height=(len(matches_data) * 38) + 50)
    st.dataframe(shots_data, use_container_width=True, height=(len(shots_data) * 38) + 50)

    # Display the date of last data update
    display_date_of_update(date_of_update)