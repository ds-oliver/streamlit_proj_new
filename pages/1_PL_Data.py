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

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, all_gws_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, style_dataframe_v2, get_color, get_color_from_palette, round_and_format, create_custom_cmap, style_dataframe_custom

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
def process_data(all_gws_data, temp_default, col_groups):
    
    df = pd.read_csv(all_gws_data)
    temp_df = pd.read_csv(temp_default)

    # capitalize the column names
    df.columns = [col.capitalize() for col in df.columns]

    df = df.merge(temp_df[['Player', 'Position', 'Team']], on='Player', suffixes=('_df', '_temp'))
    df['Position'] = df['Position_temp']
    df['Team'] = df['Team_temp']

    # drop where position is GK
    df = df[df['Position'] != 'GK']
    df = df[df['Position'].notna()]

    df.drop(columns=['Position_df', 'Team_df'], inplace=True)

    # rename 'fantrax position' column to 'position'
    # df.rename(columns={'fantrax position': 'position'}, inplace=True)

    print(df.columns.tolist())

    # create timestamp so we can use to display the date of the last data update
    date_of_update = datetime.fromtimestamp(os.path.getmtime(all_gws_data)).strftime('%d %B %Y')

    df['Games'] = df['Gameweek'].max()

    # rename Games_starts to GS, Goals_assists to G+A, any column name that starts with xg to xG
    if 'Games_starts' in df.columns:
        df.rename(columns={'Games_starts': 'GS'}, inplace=True)
    if 'Goals_assists' in df.columns:
        df.rename(columns={'Goals_assists': 'G+A'}, inplace=True)
    if 'Gameweek' in df.columns:
        df.rename(columns={'Gameweek': 'GW'}, inplace=True)

    # sort df by GS then G+A
    df.sort_values(by=['GS', 'G+A'], ascending=False, inplace=True)

    # Define default columns
    DEFAULT_COLUMNS = ['Player', 'Position', 'Team','GS']

    # rename any column name that starts with xg to xG
    df.rename(columns={col: col.replace('xg', 'xG') for col in df.columns if col.startswith('xg')}, inplace=True)

    return df, DEFAULT_COLUMNS, date_of_update, col_groups

# we want to add a date of last data update to the page
def display_date_of_update(date_of_update):
    st.sidebar.write(f'Last updated: {date_of_update}')
    

# Function to load the data
# @st.cache_resource
def load_data():
    return process_data(all_gws_data, temp_default, col_groups)

# Function to filter data based on selected Teams and positions
# @st.cache_resource
def filter_data(df, selected_Teams, selected_positions):
    return df[df['Team'].isin(selected_Teams) & df['Position'].isin(selected_positions)]

# Function to group data based on selected options
def group_data(df, selected_columns, selected_group, selected_positions, selected_Teams, grouping_option, aggregation_option):
    if grouping_option == 'Position':
        grouped_df = df.groupby('position').agg(aggregation_option).reset_index()
    elif grouping_option == 'Team':
        grouped_df = df.groupby('Team').agg(aggregation_option).reset_index()
    else:
        grouped_df = df

    columns_to_show = ['position' if grouping_option == 'Position' else 'Team'] + selected_columns if grouping_option != 'None' else selected_columns

    grouped_df = grouped_df.round(2)
    return grouped_df, columns_to_show

def get_grouping_values_and_column(grouping_option, selected_positions, selected_Teams, grouped_df, selected_stats_for_plot):
    if grouping_option == 'Position':
        return selected_positions, 'position'
    elif grouping_option == 'Team':
        return selected_Teams, 'Team'
    else:
        top_players = grouped_df.nlargest(25, selected_stats_for_plot)
        return top_players['Player'].tolist(), 'Player'

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

def create_plot(selected_group, selected_columns, selected_positions, selected_Teams, grouped_df, grouping_option):
    if selected_group and selected_columns:
        selected_stats_for_plot = st.multiselect('Select Statistics for Plotting', options=selected_columns)
        st.info('Note: If no grouping option is selected, the top 25 players by the first selected statistic is shown.')
        
        if selected_stats_for_plot:
            colors = px.colors.qualitative.Plotly[:len(selected_stats_for_plot)]
            stat_colors = {stat: color for stat, color in zip(selected_stats_for_plot, colors)}
            
            fig = go.Figure()
            grouping_values, grouping_column = get_grouping_values_and_column(grouping_option, selected_positions, selected_Teams, grouped_df, selected_stats_for_plot)
            
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

def most_recent_Team(Teams):
    return Teams.iloc[-1]

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

# def style_dataframe_custom(df, selected_columns, custom_cmap=None):
#     if custom_cmap:
#         object_cmap = custom_cmap
#     else:
#         object_cmap = create_custom_cmap() # Customized color map

#     Team_cmap = plt.cm.get_cmap('icefire')

#     styled_df = pd.DataFrame('', index=df.index, columns=df.columns)

#     position_column = 'Position' if 'Position' in df.columns else 'Position' if 'Position' in df.columns else None

#     if position_column:
#         position_colors = {
#             "D": "background-color: #6d597a",
#             "M": "background-color: #370617",
#             "F": "background-color: #03071e"
#         }
#         styled_df[position_column] = df[position_column].apply(lambda x: position_colors[x])
#         styled_df['Player'] = df[position_column].apply(lambda x: position_colors[x])

#     for col in df.columns:
#         if col in ['Player', position_column, 'Team']:
#             continue

#         unique_values = df[col].unique()
#         if len(unique_values) <= 3:  # Columns with 3 or less unique values
#             constant_colors = ["color: #eae2b7", "color: #FDFEFE", "color: #FDFAF9"] # first is slightly off-white, second is light yellow, second is pale blue
#             # You can define colors here
#             color_mapping = {val: color for val, color in zip(unique_values, constant_colors[:len(unique_values)])}
#             styled_df[col] = df[col].apply(lambda x: color_mapping[x])
#         elif 'Team' in df.columns:
#             min_val = df[col].min()
#             max_val = df[col].max()
#             range_val = float(max_val) - float(min_val)
#             styled_df[col] = df[col].astype(float).apply(lambda x: get_color((x - float(min_val)) / float(range_val), mpl_cm.get_cmap('magma')))

#         else:
#             min_val = float(df[col].min())  # Convert to float
#             max_val = float(df[col].max())  # Convert to float
#             styled_df[col] = df[col].apply(lambda x: f'color: {matplotlib.colors.to_hex(object_cmap((float(x) - min_val) / (max_val - min_val)))}' if min_val != max_val else '')

#     return styled_df

def style_dataframe_custom(df, selected_columns, custom_cmap=None):
    if custom_cmap:
        object_cmap = custom_cmap
    else:
        object_cmap = create_custom_cmap() # Customized color map

    Team_cmap = plt.cm.get_cmap('icefire')

    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)

    position_column = 'Position' if 'Position' in df.columns else 'Position' if 'Position' in df.columns else None

    if position_column:
        position_colors = {
            "D": "background-color: #6d597a",
            "M": "background-color: #370617",
            "F": "background-color: #03071e"
        }
        styled_df[position_column] = df[position_column].apply(lambda x: position_colors[x])
        styled_df['Player'] = df[position_column].apply(lambda x: position_colors[x])

    for col in df.columns:
        if col in ['Player', position_column, 'Team']:
            continue

        unique_values = df[col].unique()
        if len(unique_values) <= 3:  # Columns with 3 or less unique values
            # Columns with 3 or less unique values
            constant_colors = ["color: #eae2b7", "color: #FDFEFE", "color: #FDFAF9"]
            # Finding the most common value
            most_common_value, _ = Counter(df[col]).most_common(1)[0]
            # Assigning the rest of the colors
            other_colors = [color for val, color in zip(unique_values, constant_colors[1:]) if val != most_common_value]
            # Creating the color mapping, ensuring that most_common_value gets the first color
            color_mapping = {most_common_value: constant_colors[0], **{val: color for val, color in zip([uv for uv in unique_values if uv != most_common_value], other_colors)}}
            # Applying the color mapping, with a default value if a key is not found
            styled_df[col] = df[col].apply(lambda x: color_mapping.get(x, ''))

        elif 'Team' in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = float(max_val) - float(min_val)
            styled_df[col] = df[col].astype(float).apply(lambda x: get_color((x - float(min_val)) / float(range_val), mpl_cm.get_cmap('magma')))

        else:
            min_val = float(df[col].min())  # Convert to float
            max_val = float(df[col].max())  # Convert to float
            styled_df[col] = df[col].apply(lambda x: f'color: {matplotlib.colors.to_hex(object_cmap((float(x) - min_val) / (max_val - min_val)))}' if min_val != max_val else '')

    return styled_df

def main():
    # Load the data
    data, DEFAULT_COLUMNS, date_of_update, col_groups = load_data()

    # custom_cmap = create_custom_cmap('#03071e', '#d00000', '#f48c06')
    # custom_cmap = create_custom_cmap('#03071e', '#8d99ae', '#9C0207')
    data.head(25)

    # Display the date of last data update
    display_date_of_update(date_of_update)

    col_groups = {key.capitalize(): [col.capitalize() for col in value] for key, value in col_groups.items()}

    # Create a sidebar slider to select the GW range
    GW_range = st.slider('GW range', min_value=data['GW'].min(), max_value=data['GW'].max(), value=(data['GW'].min(), data['GW'].max()), step=1, help="Select the range of gameweeks to display data for. This slider adjusts data globally for all tables and plots")
    GW_range = list(GW_range)

    # Filter the DataFrame by the selected GW range
    data = data[(data['GW'] >= GW_range[0]) & (data['GW'] <= GW_range[1])]

    if GW_range[0] != GW_range[1]:
        selected_aggregation_method = st.sidebar.selectbox('Select Aggregation Method', ['mean', 'sum'])

        # Define aggregation functions for numeric and non-numeric columns
        aggregation_functions = {col: selected_aggregation_method if data[col].dtype in [np.float64, np.int64] else 'first' for col in data.columns}
        aggregation_functions['Player'] = 'first'
        aggregation_functions['Team'] = most_recent_Team
        aggregation_functions['Position'] = 'first' # Aggregating by the first occurrence of position
        aggregation_functions['GW'] = 'nunique' # Counting the number of GWs
        aggregation_functions['GS'] = 'sum' # Summing the number of starts

        # Group by player, Team, and position, and apply the aggregation functions
        print("Data before aggregation:", data.head())
        data = data.groupby(['Player', 'Team', 'Position'], as_index=False).agg(aggregation_functions)
        print("Data after aggregation:", data.head())
        print("Shape of matches_df after grouping by player, Team, and position:", data.shape)

        data.rename(columns={'GW': 'GP'}, inplace=True)
        data['GS:GP'] = round(data['GS'] / data['GP'].max(), 2).apply(lambda x: f"{x:.2f}")

        # Make sure 'GP' is only added once
        if 'GP' not in DEFAULT_COLUMNS:
            DEFAULT_COLUMNS.append('GP')

        # Include other necessary columns without 'GW'
        DEFAULT_COLUMNS = ['Player', 'Team', 'Position', 'GS:GP'] + [col for col in DEFAULT_COLUMNS if col not in ['Player', 'Team', 'Position', 'GS:GP', 'GW']]

        print("DEFAULT_COLUMNS:", DEFAULT_COLUMNS)

        # Sidebar filters
        selected_Teams = create_sidebar_multiselect(data, 'Team', 'Select Teams', default_all=True, key_suffix="Teams")
        selected_positions = create_sidebar_multiselect(data, 'Position', 'Select Positions', default_all=True, key_suffix="positions")

        # Filter data based on selected options
        filtered_data = filter_data(data, selected_Teams, selected_positions)

        # User selects the group and columns to show
        selected_group = st.sidebar.selectbox("Select Stats Grouping", list(col_groups.keys()))
        selected_columns = col_groups[selected_group]
        selected_columns = [col for col in selected_columns if col in data.columns]
        
        grouping_option = st.sidebar.selectbox("Select Grouping Option", ['None', 'Position', 'Team'])

        if grouping_option == 'None':
            columns_to_show = list(DEFAULT_COLUMNS) + [col for col in selected_columns if col in data.columns]
        else:
            columns_to_show = [grouping_option.capitalize()] + selected_columns

        print(columns_to_show) # Should print the list of columns you want to show

        # Group data based on selected options
        grouped_data, _ = group_data(filtered_data, selected_columns, selected_group, selected_positions, selected_Teams, grouping_option, aggregation_option=selected_aggregation_method)

        # Filter columns_to_show to include only columns that exist in grouped_data
        columns_to_show = [col for col in columns_to_show if col in grouped_data.columns]

        # Styling DataFrame
        styled_df = style_dataframe_custom(grouped_data[columns_to_show], columns_to_show, False)

        grouped_data = grouped_data.applymap(round_and_format)

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
        create_plot(selected_group, selected_columns, selected_positions, selected_Teams, grouped_data, grouping_option)


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


