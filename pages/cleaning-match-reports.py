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

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, col_groups

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

def read_data(file_path):
    return pd.read_csv(file_path)

def process_matches_data(matches_data, temp_data):
    matches_df = read_data(matches_data)
    temp_df = read_data(temp_data)
    matches_df['fantrax position'] = temp_df['Position']
    matches_df.drop(columns=['position'], inplace=True)
    matches_df.rename(columns={'fantrax position': 'position'}, inplace=True)
    return matches_df

def load_shots_data(shots_data):
    return read_data(shots_data)

def process_data():
    matches_df = process_matches_data(matches_data, temp_default)
    shots_df = load_shots_data(shots_data)
    date_of_update = datetime.fromtimestamp(os.path.getmtime(matches_data)).strftime('%d %B %Y')
    return matches_df, shots_df, date_of_update

def main():
    # Load the data
    matches_df, shots_df, DEFAULT_COLUMNS, date_of_update = load_data()

    matches_df

    shots_df

    st.dataframe(matches_df, use_container_width=True, height=(len(matches_df) * 38) + 50)
    st.dataframe(shots_df, use_container_width=True, height=(len(shots_df) * 38) + 50)

    # Display the date of last data update
    display_date_of_update(date_of_update)


if __name__ == "__main__":
    main()