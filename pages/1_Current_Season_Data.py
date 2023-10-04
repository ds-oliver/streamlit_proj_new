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
import matplotlib.cm as mpl_cm
from pandas.io.formats.style import Styler
import cProfile
import pstats
import io
import matplotlib.colors as mcolors
import matplotlib
from collections import Counter
from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import percentileofscore

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_col_groups, matches_drop_cols, matches_default_cols, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, matches_default_cols_rename, matches_standard_cols_rename, matches_defense_cols_rename, matches_passing_cols_rename, matches_possession_cols_rename, matches_misc_cols_rename, matches_pass_types_rename, colors, divergent_colors, matches_rename_dict, colors, divergent_colors, matches_rename_dict

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, all_gws_data, pl_2018_2023, matches_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, style_dataframe_v2, get_color, get_color_from_palette, round_and_format, create_custom_cmap, style_dataframe_custom, add_construction, display_date_of_update, load_css, create_custom_sequential_cmap, rank_players_by_multiple_stats, percentile_players_by_multiple_stats, debug_dataframe

st.set_page_config(
    page_title="Footy Magic",
    page_icon=":soccer:",
    layout="wide",  
    initial_sidebar_state="expanded",
    menu_items={

    }
)

# Load the CSS file
# In pages/script.py
load_css()

# scraped data from : /Users/hogan/dev/fbref/scripts/rfx_scrape/fbref-scrape-current-year.py

# logger = st.logger

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

# print("Scripts path:", scripts_path)

# print(sys.path)

# @st.cache_data
# Function to create rename dictionary

# Function to filter data based on selected Teams and positions
# @st.cache_data
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

def create_plot(selected_group, selected_columns, selected_positions, selected_Team, grouped_df, grouping_option):
    if selected_group and selected_columns:
        selected_stats_for_plot = st.multiselect('Select Statistics for Plotting', options=selected_columns)
        st.info('Note: If no grouping option is selected, the top 25 players by the first selected statistic is shown.')
        
        if selected_stats_for_plot:
            colors = px.colors.qualitative.Plotly[:len(selected_stats_for_plot)]
            stat_colors = {stat: color for stat, color in zip(selected_stats_for_plot, colors)}
            
            fig = go.Figure()

            # Filter the grouped_df based on selected_Team if it's not 'All Teams'
            if selected_Team != 'All Teams':
                grouped_df = grouped_df[grouped_df['Team'] == selected_Team]

            grouping_values, grouping_column = get_grouping_values_and_column(grouping_option, selected_positions, selected_Team, grouped_df, selected_stats_for_plot)

            add_bar_traces(fig, selected_stats_for_plot, grouping_values, grouped_df, grouping_column, stat_colors)
            
            if grouping_option == 'None':
                title = f'Plotting of Players for Selected Statistics'
            else:
                title = f'Plotting of Selected {grouping_option} for Selected Statistics'

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

def most_recent_Team(Teams):
    return Teams.iloc[-1]

def create_pivot_table(data, DEFAULT_COLUMNS, matches_col_groups):
    # Debugging
    st.write(f"Data Shape: {data.shape}")
    st.write(f"Data Columns: {data.columns.tolist()}")
    st.write(f"Data Head: {data.head()}")
    st.write(f"Data Types: {data.dtypes}")
    
    # Ensure 'Player' column is of type string
    data['Player'] = data['Player'].astype(str)
    
    unique_players = data['Player'].unique()
    st.write(f"Unique values in Player: {unique_players}")

    pivot_index = st.sidebar.selectbox('Select Index for Pivot Table', DEFAULT_COLUMNS)
    selected_group = st.sidebar.selectbox('Select Column Group for Pivot Table', list(matches_col_groups.keys()))
    pivot_values = st.sidebar.selectbox('Select Values for Pivot Table', DEFAULT_COLUMNS)
    pivot_agg_func = st.sidebar.selectbox('Select Aggregation Function for Pivot Table', ['mean', 'sum', 'count', 'min', 'max'])

    selected_columns = matches_col_groups[selected_group]
    selected_columns = [col for col in selected_columns if col in data.columns]

    try:
        pivot_table = pd.pivot_table(data, values=pivot_values, index=pivot_index, columns=selected_columns, aggfunc=pivot_agg_func)
        st.write(f"Pivot Table by {pivot_index}, {selected_columns}, and {pivot_values} with {pivot_agg_func} aggregation")
        st.write(pivot_table)
    except Exception as e:
        st.warning(f"An error occurred while creating the pivot table: {e}")

def create_multi_index(data, DEFAULT_COLUMNS):
    index_level_1 = st.sidebar.selectbox('Select First Index for Multi-level', DEFAULT_COLUMNS)
    index_level_2 = st.sidebar.selectbox('Select Second Index for Multi-level', DEFAULT_COLUMNS)
    
    multi_index_df = data.set_index([index_level_1, index_level_2])
    st.write(f"DataFrame with Multi-level Indexing by {index_level_1} and {index_level_2}")
    st.write(multi_index_df)

def format_col_names(df, default_columns):
    # do not format the default columns
    # format the rest of the columns

    df.rename(columns={col: col.replace('_', ' ').title() for col in df.columns if col not in default_columns}, inplace=True)
    return df

# def style_dataframe_custom(df, selected_columns, custom_cmap="copper"):
#     object_cmap = plt.cm.get_cmap(custom_cmap)
#     styled_df = pd.DataFrame()

#     position_column = 'Position' if 'Position' in df.columns else None
#     if position_column:
#         position_colors = {
#             "D": "background-color: #6d597a; color: white",
#             "M": "background-color: #370617; color: white",
#             "F": "background-color: #03071e; color: white"
#         }
#         styled_df[position_column] = df[position_column].apply(lambda x: position_colors.get(x, ''))

#         if 'Player' in df.columns:
#             styled_df['Player'] = df[position_column].apply(lambda x: position_colors.get(x, ''))

#     for col in selected_columns:
#         if col in ['Player', position_column]:
#             continue

#         col_data = df[col]

#         try:
#             col_data = col_data.astype(float)
#             min_val = col_data.min()
#             max_val = col_data.max()
#         except ValueError:
#             min_val = max_val = None

#         unique_values = col_data.unique()

#         if len(unique_values) <= 3:
#             constant_colors = ["#060301", "#6d0301", "#FDFAF9"]

#             most_common_list = Counter(col_data).most_common(1)
#             if most_common_list:
#                 most_common_value, _ = most_common_list[0]
#             else:
#                 most_common_value = None

#             other_values = [uv for uv in unique_values if uv != most_common_value]
#             text_colors = ['black' if color == "#FDFAF9" else 'white' for color in constant_colors]

#             color_mapping = {
#                 val: f"background-color: {color}; color: {text}" 
#                 for val, color, text in zip([most_common_value] + other_values, constant_colors, text_colors)
#             }

#             styled_df[col] = col_data.apply(lambda x: color_mapping.get(x, ''))
#         elif min_val is not None and max_val is not None:
#             if min_val != max_val:
#                 styled_df[col] = col_data.apply(
#                     lambda x: get_color((x - min_val) / (max_val - min_val), object_cmap)
#                 )
#     return styled_df

# Function to group data based on selected options
def group_data(df, selected_columns, grouping_option, aggregation_option='sum', exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
        
    # Convert selected_columns to numeric type before aggregation, skipping those in exclude_cols
    for col in selected_columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    aggregation_methods = {col: aggregation_option for col in selected_columns if df[col].dtype in [np.float64, np.int64]}
    
    if grouping_option == 'Position':
        grouped_df = df.groupby('Position').agg(aggregation_methods).reset_index()
    elif grouping_option == 'Team':
        grouped_df = df.groupby('Team').agg(aggregation_methods).reset_index()
    else:
        grouped_df = df
        
    return grouped_df

def rename_columns(df, rename_dict, col_groups):
    print("Debug: Beginning of rename_columns function")
    print(f"Debug: rename_dict value is: {rename_dict}")

    # Capitalize the keys and values in rename_dict
    rename_dict = {k.capitalize(): v for k, v in rename_dict.items()}

    # Identify columns that are in the DataFrame but not in the rename_dict (and vice versa)
    cols_lost = set(df.columns) - set(rename_dict.keys())
    new_cols = set(rename_dict.values()) - set(df.columns)
    print(f"\nDebug: Columns lost: {cols_lost} \n\nNew columns: {new_cols}")

    # Update the column groups before renaming DataFrame columns
    updated_col_groups = {}
    for group, cols in col_groups.items():
        updated_col_groups[group] = [rename_dict.get(col.capitalize(), col) for col in cols]

    # Rename columns in the DataFrame
    df.rename(columns=rename_dict, inplace=True)

    return df, len(rename_dict), updated_col_groups

def get_grouping_values_and_column(grouping_option, selected_positions, selected_Teams, grouped_df, selected_stats_for_plot):
    grouped_df[selected_stats_for_plot] = grouped_df[selected_stats_for_plot].apply(pd.to_numeric, errors='coerce')

    if grouping_option == 'Position':
        return selected_positions, 'Position'
    elif grouping_option == 'Team':
        return selected_Teams, 'Team'
    else:
        # convert the selected_stats_for_plot datatypes to numeric
        top_players = grouped_df.nlargest(25, selected_stats_for_plot)
        return top_players['Player'].tolist(), 'Player'

# @st.cache_data
def filter_data(df, selected_Team, selected_positions):
    print("Debug from inside filter_data function: selected_Team and selected_positions are: ", selected_Team, selected_positions)
    
    # Filter by Team if not 'All Teams'
    if selected_Team != 'All Teams':
        df = df[df['Team'] == selected_Team]
        print("Debug from inside filter_data function: Shape of df after filtering by Team:", df.shape)
    
    # Filter by Position only if selected_positions is not empty
    if selected_positions:
        df = df[df['Position'].isin(selected_positions)]
    
    # Print shape of df after filtering
    print("Debug from inside filter_data function: Shape of df after filtering by Team and then by Position:", df.shape)
    
    return df

def ensure_unique_columns(df):
    unique_cols = []
    seen = set()
    for col in df.columns:
        unique_col = col
        count = 1
        while unique_col in seen:
            unique_col = f"{col}_{count}"
            count += 1
        seen.add(unique_col)
        unique_cols.append(unique_col)
    df.columns = unique_cols

def create_rename_dict(old_cols_list, new_cols_list):
    if len(old_cols_list) != len(new_cols_list):
        raise ValueError("The length of old_cols_list must be the same as new_cols_list.")
    rename_dict = dict(zip(old_cols_list, new_cols_list))
    return rename_dict

# Function to process data
@st.cache_data
def process_data(matches_data, temp_default, matches_drop_cols, matches_default_cols):
    df = pd.read_csv(matches_data)
    temp_df = pd.read_csv(temp_default)
    
    # Merging DataFrames
    df = pd.merge(df, temp_df[['Player', 'Position', 'Team']], left_on='player', right_on='Player', how='left')
    
    # Drop unnecessary columns
    df.drop(columns=['Player', 'team'], inplace=True)
    
    # Filter out rows where Position is 'GK' or NaN
    df = df[df['Position'] != 'GK']
    df = df[df['Position'].notna()]

    # Rename 'gameweek' to 'GW'
    df.rename(columns={'gameweek': 'GW'}, inplace=True)
    
    # Drop columns specified in matches_drop_cols
    for col in matches_drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(subset=['player', 'GW'], inplace=True)

    # Combine all rename dictionaries into one
    all_rename_dicts = {**matches_default_cols_rename, **matches_standard_cols_rename, **matches_passing_cols_rename,**matches_pass_types_rename, **matches_defense_cols_rename, **matches_possession_cols_rename, **matches_misc_cols_rename}
    
    # Rename columns based on the combined dictionary
    df.rename(columns=all_rename_dicts, inplace=True)

    # Handle x-prefixed columns
    # df.rename(columns={col: col.replace('X', 'x') for col in df.columns if col.startswith('X')}, inplace=True)
    # df.rename(columns={col: col[:2] + col[2:].capitalize() if col.startswith('x') and len(col) == 2 else col[:2] + col[2:4].capitalize() + col[4:] if col.startswith('x') and len(col) == 3 else col for col in df.columns if col.startswith('x')}, inplace=True)

    # Update MATCHES_DEFAULT_COLS based on matches_default_cols
    MATCHES_DEFAULT_COLS = [col if col != 'GW' else col for col in matches_default_cols]
    
    # Get the date of last update
    date_of_update = datetime.fromtimestamp(os.path.getmtime(matches_data)).strftime('%d %B %Y')
    
    return df, MATCHES_DEFAULT_COLS, date_of_update
 
# Function to load the data
@st.cache_data
def load_data():
    return process_data(matches_data, temp_default, matches_drop_cols, matches_default_cols)


def main():
    add_construction()

    print("Debug: Beginning of main function")
    # print timestamp
    print("Timestamp:", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    custom_cmap = create_custom_sequential_cmap(*colors)
    custom_divergent_cmap = create_custom_sequential_cmap(*divergent_colors)

    # Updating the column groups to use renamed columns
    matches_col_groups = {
        "Standard": list(matches_standard_cols_rename.values()),
        "Passing": list(matches_passing_cols_rename.values()),
        "Defense": list(matches_defense_cols_rename.values()),
        "Possession": list(matches_possession_cols_rename.values()),
        "Miscellaneous": list(matches_misc_cols_rename.values()),
        "Passing Types": list(matches_pass_types_rename.values()),
    }

    data, DEFAULT_COLUMNS, date_of_update = load_data()
    debug_dataframe(data, "Debugging initial data (DataFrame: data)")

    display_date_of_update(date_of_update)

    column_rename_dict = {'Gw': 'GW', 'Started': 'GS'}
    data.rename(columns=column_rename_dict, inplace=True)

    GW_range = st.slider('GW range', min_value=int(data['GW'].min()), max_value=int(data['GW'].max()), value=(int(data['GW'].min()), int(data['GW'].max())), step=1)
    GW_range = list(GW_range)

    data = data[(data['GW'] >= GW_range[0]) & (data['GW'] <= GW_range[1])]
    debug_dataframe(data, "Debugging data after filtering by GW range (DataFrame: data)")

    exclude_cols = ['Player', 'Team', 'Position', 'Nation', 'Season']
    for col in data.columns:
        if col not in exclude_cols:
            if pd.api.types.is_object_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')

    selected_aggregation_method = st.sidebar.selectbox('Select Aggregation Method', ['Mean', 'Sum'])
    selected_aggregation_method = selected_aggregation_method.lower()
    aggregation_functions = {col: selected_aggregation_method if pd.api.types.is_numeric_dtype(data[col]) else 'first' for col in data.columns}

    aggregation_functions['Player'] = 'first'
    aggregation_functions['Team'] = most_recent_Team
    aggregation_functions['Position'] = 'first'
    aggregation_functions['GW'] = 'nunique'
    aggregation_functions['GS'] = 'sum'

    data = data.groupby(['Player', 'Team', 'Position'], as_index=False).agg(aggregation_functions)
    data.rename(columns={'GW': 'GP'}, inplace=True)
    data['GS:GP'] = round(data['GS'] / data['GP'].max(), 2).apply(lambda x: f"{x:.2f}")

    if 'GP' not in DEFAULT_COLUMNS:
        DEFAULT_COLUMNS.append('GP')

    DEFAULT_COLUMNS = ['Player', 'Team', 'Position', 'GS:GP'] + [col for col in DEFAULT_COLUMNS if col not in ['Player', 'Team', 'Position', 'GS:GP', 'GW']]

    all_teams = data['Team'].unique().tolist()
    all_teams.sort()
    all_teams = ['All Teams'] + all_teams
    selected_Team = st.sidebar.selectbox('Select Team', all_teams)

    all_positions = data['Position'].unique().tolist()
    selected_positions = create_sidebar_multiselect(data, 'Position', 'Select Positions', default_all=True)

    filtered_data = filter_data(data, selected_Team, selected_positions)
    debug_dataframe(filtered_data, "Debugging data after team and position filtering (DataFrame: filtered_data)")

    selected_group = st.sidebar.selectbox("Select Stats Grouping", list(matches_col_groups.keys()))
    selected_columns = matches_col_groups[selected_group]
    selected_columns = [col for col in selected_columns if col in data.columns]

    columns_to_show = list(DEFAULT_COLUMNS) + selected_columns
    columns_to_show = [col for col in columns_to_show if col in filtered_data.columns]

    show_as_rank = st.sidebar.radio('Show stats values as:', ['Original Values', 'Relative Percentile'])
    grouping_option = st.sidebar.selectbox("Select Grouping Option", ['None', 'Position', 'Team'])

    if show_as_rank == 'Relative Percentile':
        grouped_data = percentile_players_by_multiple_stats(filtered_data, selected_columns)
    else:
        grouped_data = filtered_data

    if grouping_option != 'None':
        grouped_data = group_data(grouped_data, selected_columns, grouping_option, exclude_cols=exclude_cols)
    else:
        set_index_to_player = st.sidebar.checkbox('Set index to Player', False)
        if set_index_to_player:
            grouped_data.set_index('Player', inplace=True)

    ensure_unique_columns(grouped_data)

    grouped_data = grouped_data.applymap(round_and_format)
    debug_dataframe(grouped_data, "Debugging grouped data after applymap (DataFrame: grouped_data)")

    final_cmap = custom_divergent_cmap if show_as_rank == 'Relative Percentile' else custom_cmap
    is_percentile = (show_as_rank == 'Relative Percentile')

    styled_df = style_dataframe_custom(grouped_data[columns_to_show], columns_to_show, custom_cmap=final_cmap, inverse_cmap=False, is_percentile=is_percentile)

    st.header(f"Premier League Players' Statistics ({selected_group})")

    filtered_df = dataframe_explorer(grouped_data[columns_to_show])
    filtered_df.reset_index(drop=True, inplace=True)

    st.dataframe(
        filtered_df.style.apply(lambda _: styled_df, axis=None),
        use_container_width=True,
        height=(len(filtered_df) * 30) + 50 if grouping_option != 'None' else 35 * 20
    )
        
    create_plot(selected_group, selected_columns, selected_positions, selected_Team, grouped_data, grouping_option)


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()

    main() # This calls your main function

    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()

    # with open('profile_output.txt', 'w') as f:
    #     f.write(s.getvalue())


