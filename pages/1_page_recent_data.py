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

# Adding path to the scripts directory
scripts_path = os.path.abspath(os.path.join('./scripts'))
sys.path.append(scripts_path)

# Adding path to the directory containing constants.py
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

st.set_page_config(
    layout="wide"
)

from constants import fbref_cats, fbref_leagues, seasons, stats_cols, shooting_cols, passing_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, passing_types_cols, gca_cols, color1, color2, color3, color4, color5, cm, big5_players_csv, fbref_base_url, fbref_current_year_url

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect


# Read the data
df = pd.read_csv(pl_data_gw1)

temp_df = pd.read_csv(temp_default)

temp_cols = temp_df.columns.tolist()

df['fantrax position'] = temp_df['Position']

# drop df['position'] column
df.drop(columns=['position'], inplace=True)

# # Filter data for rows where Comp is 'eng Premier League'
# df = df[df['comp_level'] == 'eng Premier League']

# # apply clean_age_column function to 'age' column
# df = clean_age_column(df)

# # Drop 'comp_level', 'nationality', 'birth_year' columns
# drop_cols = ['ranker', 'nationality', 'birth_year', 'comp_level']
# df.drop(columns=drop_cols, inplace=True)

# # Define default columns
DEFAULT_COLUMNS = ['player', 'fantrax position', 'team', 'games_starts']

# Exclude the default columns
stat_cols = [col for col in df.columns if col not in DEFAULT_COLUMNS]

# create a multiselect for the teams, default to all teams
selected_teams = create_sidebar_multiselect(df, 'team', 'Select Teams', default_all=True)

# Filter the DataFrame for selected teams
df = df[df['team'].isin(selected_teams)]

# if there is no team selected, display a message
if len(selected_teams) == 0:
    st.write('Please select at least one team.')

#create a multiselect for the positions, default to all positions
selected_positions = create_sidebar_multiselect(df, 'fantrax position', 'Select Positions', default_all=True)

col_groups = {
    "Standard": stats_cols,
    "Shooting": shooting_cols,
    "Passing": passing_cols,
    "Defense": defense_cols,
    "Possession": possession_cols,
    "Miscellaneous": misc_cols,
    "Passing Types": passing_types_cols,
    "GCA": gca_cols,
    "Playing Time": playing_time_cols,
}

selected_group = st.sidebar.selectbox('Select a Category', options=list(col_groups.keys()))
selected_columns = col_groups[selected_group]

grouping_option = st.radio(
    'Group Data by:', ('None', 'Position', 'Team')
)

# Offer radio buttons for different aggregation options
aggregation_option = st.radio(
    'Select Aggregation Option:', ('Mean', 'Median', 'Sum')
)

# Determine the aggregation function based on the selected option
if aggregation_option == 'Sum':
    aggregation_func = 'sum'
elif aggregation_option == 'Mean':
    aggregation_func = 'mean'
elif aggregation_option == 'Median':
    aggregation_func = 'median'

# Group the DataFrame based on the selected option
if grouping_option == 'Position':
    grouped_df = df.groupby('fantrax position').agg(aggregation_func).reset_index()
    # round to 2 decimal places for all columns
    grouped_df = grouped_df.round(2)
    columns_to_show = ['fantrax position'] + selected_columns
    st.dataframe(grouped_df[columns_to_show], use_container_width=True, height=len(df['fantrax position'].unique())*50)
elif grouping_option == 'Team':
    grouped_df = df.groupby('team').agg(aggregation_func).reset_index()
    columns_to_show = ['team'] + selected_columns
    grouped_df = grouped_df.round(2)
    st.dataframe(grouped_df[columns_to_show], use_container_width=True, height=len(df['team'].unique())*37)
else:
    grouped_df = df
    columns_to_show = DEFAULT_COLUMNS + selected_columns
    st.dataframe(grouped_df[columns_to_show], use_container_width=True, height=1000)


# Check if there are selected groups and columns
if selected_group and selected_columns:
    selected_stats_for_plot = st.multiselect('Select Statistics for Plotting', options=selected_columns)

    if selected_stats_for_plot:
        # Define colors for statistics
        colors = px.colors.qualitative.Plotly[:len(selected_stats_for_plot)]
        stat_colors = {stat: color for stat, color in zip(selected_stats_for_plot, colors)}

        fig = go.Figure()

        if grouping_option == 'Position':
            grouping_values = selected_positions
            grouping_column = 'fantrax position'
        elif grouping_option == 'Team':
            grouping_values = selected_teams
            grouping_column = 'team'
        else:
            grouping_values = sorted(df['player'].unique())
            grouping_column = 'player'

        # Iterate through selected statistics and add trace for each grouping value
        for stat in selected_stats_for_plot:
            x_values = []
            y_values = []
            for value in grouping_values:
                x_values.append(value)
                y_values.append(grouped_df[grouped_df[grouping_column] == value][stat].values[0])
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    name=stat,
                    text=y_values,
                    textposition='outside',
                    marker_color=stat_colors[stat]  # Use statistic-specific color
                )
            )

        title = f'Comparison of Selected {grouping_option} for Selected Statistics'
        fig.update_layout(
            title=title,
            xaxis_title=grouping_option if grouping_option != 'None' else 'Players',
            yaxis_title='Value',
            legend=dict(
                bgcolor='rgba(255,255,255,0.5)',
                font=dict(
                    color='black'
                )
            ),
            barmode='group',  # This groups the bars for each grouping value together
            height=500  # Set the height of the plot
        )
        fig.update_traces(hoverinfo="x+y+name")  # Show hover information
        st.plotly_chart(fig, use_container_width=True)






