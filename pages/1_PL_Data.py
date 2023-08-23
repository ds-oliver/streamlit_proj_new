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

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues

print("Scripts path:", scripts_path)

print(sys.path)

st.set_page_config(
    layout="wide"
)

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, all_gws_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, style_dataframe_v2, get_color as get_color_v2, get_color_from_palette

# def get_color(value, cmap):
#     color_fraction = value
#     rgba_color = cmap(color_fraction)
#     brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
#     text_color = 'white' if brightness < 0.7 else 'black'
#     return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

# def style_dataframe(df, selected_columns):
#     cm_coolwarm = cm.get_cmap('inferno')
#     object_cmap = cm.get_cmap('gnuplot2')  # Choose a colormap for object columns

#     # Create an empty DataFrame with the same shape as df
#     styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
#     for col in df.columns:
#         if col == 'player':  # Skip the styling for the 'player' column
#             continue
#         if df[col].dtype in [np.float64, np.int64] and col in selected_columns:
#             min_val = df[col].min()
#             max_val = df[col].max()
#             range_val = max_val - min_val
#             styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / range_val, cm_coolwarm))
#         elif df[col].dtype == 'object':
#             unique_values = df[col].unique().tolist()
#             styled_df[col] = df[col].apply(lambda x: get_color(unique_values.index(x) / len(unique_values), object_cmap))
#     return styled_df

@st.cache_resource
def process_data(pl_data_gw1, temp_default):
    
    df = pd.read_csv(pl_data_gw1)
    temp_df = pd.read_csv(temp_default)
    df['fantrax position'] = temp_df['Position']

    # drop df['position'] column
    df.drop(columns=['position'], inplace=True)

    # rename 'fantrax position' column to 'position'
    df.rename(columns={'fantrax position': 'position'}, inplace=True)

    # Define default columns
    DEFAULT_COLUMNS = ['player', 'position', 'team', 'games_starts']

    # create timestamp so we can use to display the date of the last data update
    date_of_update = datetime.fromtimestamp(os.path.getmtime(pl_data_gw1)).strftime('%d %B %Y')

    return df, DEFAULT_COLUMNS, date_of_update

# we want to add a date of last data update to the page
def display_date_of_update(date_of_update):
    st.sidebar.write(f'Last updated: {date_of_update}')
    

# Function to load the data
@st.cache_resource
def load_data():
    return process_data(pl_data_gw1, temp_default)

# Function to filter data based on selected teams and positions
@st.cache_resource
def filter_data(df, selected_teams, selected_positions):
    return df[df['team'].isin(selected_teams) & df['position'].isin(selected_positions)]

# Function to group data based on selected options
def group_data(df, selected_columns, selected_group, selected_positions, selected_teams, grouping_option, aggregation_option):
    if grouping_option == 'Position':
        grouped_df = df.groupby('position').agg(aggregation_option).reset_index()
    elif grouping_option == 'Team':
        grouped_df = df.groupby('team').agg(aggregation_option).reset_index()
    else:
        grouped_df = df

    columns_to_show = ['position' if grouping_option == 'Position' else 'team'] + selected_columns if grouping_option != 'None' else selected_columns

    grouped_df = grouped_df.round(2)
    return grouped_df, columns_to_show

def get_grouping_values_and_column(grouping_option, selected_positions, selected_teams, grouped_df, selected_stats_for_plot):
    if grouping_option == 'Position':
        return selected_positions, 'position'
    elif grouping_option == 'Team':
        return selected_teams, 'team'
    else:
        top_players = grouped_df.nlargest(25, selected_stats_for_plot)
        return top_players['player'].tolist(), 'player'

def add_bar_traces(fig, selected_stats_for_plot, grouping_values, grouped_df, grouping_column, stat_colors):
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
                marker_color=stat_colors[stat]
            )
        )

def create_plot(selected_group, selected_columns, selected_positions, selected_teams, grouped_df, grouping_option):
    if selected_group and selected_columns:
        selected_stats_for_plot = st.multiselect('Select Statistics for Plotting', options=selected_columns)
        st.info('Note: If no grouping option is selected, the top 25 players by the first selected statistic is shown.')
        
        if selected_stats_for_plot:
            colors = px.colors.qualitative.Plotly[:len(selected_stats_for_plot)]
            stat_colors = {stat: color for stat, color in zip(selected_stats_for_plot, colors)}
            
            fig = go.Figure()
            grouping_values, grouping_column = get_grouping_values_and_column(grouping_option, selected_positions, selected_teams, grouped_df, selected_stats_for_plot)
            
            add_bar_traces(fig, selected_stats_for_plot, grouping_values, grouped_df, grouping_column, stat_colors)
            
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
                barmode='group',
                height=500
            )
            fig.update_traces(hoverinfo="x+y+name")
            st.plotly_chart(fig, use_container_width=True)

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

def main():
    # Load the data
    data, DEFAULT_COLUMNS, date_of_update = load_data()

    # Display the date of last data update
    display_date_of_update(date_of_update)

    # Sidebar filters
    selected_teams = create_sidebar_multiselect(data, 'team', 'Select Teams', default_all=True, key_suffix="teams")
    selected_positions = create_sidebar_multiselect(data, 'position', 'Select Positions', default_all=True, key_suffix="positions")

    # Filter data based on selected options
    filtered_data = filter_data(data, selected_teams, selected_positions)

    # User selects the group and columns to show
    selected_group = st.sidebar.selectbox("Select Stats Grouping", list(col_groups.keys()))
    selected_columns = col_groups[selected_group]
    
    grouping_option = st.sidebar.selectbox("Select Grouping Option", ['None', 'Position', 'Team'])

    if grouping_option == 'None':
        columns_to_show = DEFAULT_COLUMNS + selected_columns
    else:
        columns_to_show = [grouping_option.lower()] + selected_columns

    # Group data based on selected options
    grouped_data, _ = group_data(filtered_data, selected_columns, selected_group, selected_positions, selected_teams, grouping_option, aggregation_option='mean')
    
    # Styling DataFrame
    styled_df = style_dataframe_v2(grouped_data[columns_to_show], selected_columns=selected_columns)

    # Display the DataFrame
    # if grouping_option == 'None' then set st.dataframe() height= to the height of showing the first 50 rows, else set height to length of grouped_data
    if grouping_option == 'None':
        # state at the top of the page as header the grouping option selected
        st.header(f"Premier League Individual Players' Statistics:{selected_group}")
        st.dataframe(grouped_data[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=50 * 20)
    else:
        # state at the top of the page as header the grouping option selected
        st.header(f"Premier League Players' Statistics grouped by:{selected_group}")
        st.dataframe(grouped_data[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=(len(grouped_data) * 38) + 50)

    # Create plot
    create_plot(selected_group, selected_columns, selected_positions, selected_teams, grouped_data, grouping_option)

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    main() # This calls your main function

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open('profile_output.txt', 'w') as f:
        f.write(s.getvalue())


