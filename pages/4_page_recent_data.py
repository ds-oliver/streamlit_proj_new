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
import plotly.graph_objects as go
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join('./scripts')))

from constants import color1, color2, color3, color4, color5, cm, fbref_leagues as leagues, fbref_cats as cats, seasons, color_dark1, fbref_base_url

from files import big5_this_year # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column

# Read the data
df = pd.read_csv(big5_this_year)

# Filter data for rows where Comp is 'eng Premier League'
df = df[df['comp_level'] == 'eng Premier League']

# Drop 'comp_level', 'nationality', 'birth_year' columns
drop_cols = ['ranker', 'nationality', 'birth_year', 'comp_level']
df.drop(columns=drop_cols, inplace=True)

# Define default columns
DEFAULT_COLUMNS = ['player', 'position', 'team', 'games_starts']

# Exclude the default columns
stat_cols = [col for col in df.columns if col not in DEFAULT_COLUMNS]

# Sidebar multiselect for statistical categories
selected_stats = st.sidebar.multiselect(
    'Select Statistical Categories', options=stat_cols, default=stat_cols
)

# Add default columns to the selected statistical categories
columns_to_show = DEFAULT_COLUMNS + selected_stats

# Display the DataFrame
st.dataframe(df[columns_to_show])

# Optionally, if you want to add some plotting with selected stats:
if len(selected_stats) > 1:
    selected_player = st.selectbox('Select Player for Plotting', sorted(df['player'].unique()))
    selected_player_data = df[df['player'] == selected_player][selected_stats]
    
    # For example, plotting selected stats for a selected player using Plotly Express
    fig = px.bar(selected_player_data, x=selected_stats, y=selected_player_data.values[0])
    st.plotly_chart(fig)
