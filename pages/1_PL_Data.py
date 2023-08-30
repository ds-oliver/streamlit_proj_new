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
# from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
# from streamlit_extras.metric_cards import style_metric_cards
# from streamlit_extras.stylable_container import stylable_container



# scraped data from : /Users/hogan/dev/fbref/scripts/rfx_scrape/fbref-scrape-current-year.py

# logger = st.logger

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

print("Scripts path:", scripts_path)

print(sys.path)

st.set_page_config(
    layout="wide"
)

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, col_groups

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, all_gws_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, style_dataframe_v2, get_color, get_color_from_palette, round_and_format, create_custom_cmap, style_dataframe_custom, add_construction, display_date_of_update

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

@st.cache_data
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

    df.drop(columns=['Position_df', 'Team_df', 'Age'], inplace=True)

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
    df.rename(columns={col: col.replace('Xg', 'xG') for col in df.columns if col.startswith('Xg')}, inplace=True)

    return df, DEFAULT_COLUMNS, date_of_update, col_groups
    
# Function to load the data
# @st.cache_data
def load_data():
    return process_data(all_gws_data, temp_default, col_groups)

# Function to filter data based on selected Teams and positions
# @st.cache_data
def filter_data(df, selected_Teams, selected_positions):
    return df[df['Team'].isin(selected_Teams) & df['Position'].isin(selected_positions)]

# Function to group data based on selected options
def group_data(df, selected_columns, grouping_option, aggregation_option='sum'):
    aggregation_methods = {col: aggregation_option for col in selected_columns if df[col].dtype in [np.float64, np.int64]}
    if grouping_option == 'Position':
        grouped_df = df.groupby('Position').agg(aggregation_methods).reset_index()
    elif grouping_option == 'Team':
        grouped_df = df.groupby('Team').agg(aggregation_methods).reset_index()
    else:
        grouped_df = df
        
    return grouped_df

def get_grouping_values_and_column(grouping_option, selected_positions, selected_Teams, grouped_df, selected_stats_for_plot):
    grouped_df[selected_stats_for_plot] = grouped_df[selected_stats_for_plot].apply(pd.to_numeric, errors='coerce')

    if grouping_option == 'Position':
        return selected_positions, 'Position'
    elif grouping_option == 'Team':
        return selected_Teams, 'Team'
    else:
        # convert the selected_stats_for_plot datatypes to numeric
        # grouped_df[selected_stats_for_plot] = grouped_df[selected_stats_for_plot].apply(pd.to_numeric, errors='coerce')
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

def create_pivot_table(data, DEFAULT_COLUMNS, col_groups):
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
    selected_group = st.sidebar.selectbox('Select Column Group for Pivot Table', list(col_groups.keys()))
    pivot_values = st.sidebar.selectbox('Select Values for Pivot Table', DEFAULT_COLUMNS)
    pivot_agg_func = st.sidebar.selectbox('Select Aggregation Function for Pivot Table', ['mean', 'sum', 'count', 'min', 'max'])

    selected_columns = col_groups[selected_group]
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

def format_col_names(df, default_columns):
    # do not format the default columns
    # format the rest of the columns

    df.rename(columns={col: col.replace('_', ' ').title() for col in df.columns if col not in default_columns}, inplace=True)
    return df

def style_dataframe_custom(df, selected_columns, custom_cmap=None):
    if custom_cmap:
        object_cmap = custom_cmap
    else:
        object_cmap = create_custom_cmap() # Customized color map

    Team_cmap = plt.cm.get_cmap('magma')

    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)

    position_column = 'Position' if 'Position' in df.columns else 'Position' if 'Position' in df.columns else None

    if position_column:
        position_colors = {
            "D": "background-color: #6d597a",
            "M": "background-color: #370617",
            "F": "background-color: #03071e"
        }
        styled_df[position_column] = df[position_column].apply(lambda x: position_colors[x])
        if 'Player' in df.columns:
            styled_df['Player'] = df[position_column].apply(lambda x: position_colors[x])

    for col in df.columns:
        if col in ['Player', position_column]:
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
            n = len(unique_values)
            for i, val in enumerate(unique_values):
                norm_i = i / (n - 1) if n > 1 else 0.5  # Avoid division by zero
                styled_df.loc[df[col] == val, col] = get_color(norm_i, Team_cmap)


        else:
            min_val = float(df[col].min())  # Convert to float
            max_val = float(df[col].max())  # Convert to float
            styled_df[col] = df[col].apply(lambda x: f'color: {matplotlib.colors.to_hex(object_cmap((float(x) - min_val) / (max_val - min_val)))}' if min_val != max_val else '')

    return styled_df

def main():
    
    add_construction()

    # Load the data
    data, DEFAULT_COLUMNS, date_of_update, col_groups = load_data()

    # Display the date of last data update
    display_date_of_update(date_of_update)

    # Create a sidebar slider to select the GW range
    GW_range = st.slider('GW range', min_value=data['GW'].min(), max_value=data['GW'].max(), value=(data['GW'].min(), data['GW'].max()), step=1, help="Select the range of gameweeks to display data for. This slider adjusts data globally for all tables and plots", key="GW_range")
    GW_range = list(GW_range)

    # Filter the DataFrame by the selected GW range
    data = data[(data['GW'] >= GW_range[0]) & (data['GW'] <= GW_range[1])]

    if GW_range[0] != GW_range[1]:
        selected_aggregation_method = st.sidebar.selectbox('Select Aggregation Method', ['mean', 'sum'], key="aggregation_method")

        aggregation_functions = {col: selected_aggregation_method if data[col].dtype in [np.float64, np.int64] else 'first' for col in data.columns}
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

    selected_Teams = create_sidebar_multiselect(data, 'Team', 'Select Teams', default_all=True, key_suffix="teams")
    selected_positions = create_sidebar_multiselect(data, 'Position', 'Select Positions', default_all=True, key_suffix="positions")

    filtered_data = filter_data(data, selected_Teams, selected_positions)

    col_groups = {key.capitalize(): [col.capitalize() for col in value] for key, value in col_groups.items()}
    selected_group = st.sidebar.selectbox("Select Stats Grouping", list(col_groups.keys()))
    selected_columns = col_groups[selected_group]
    selected_columns = [col for col in selected_columns if col in data.columns]

    grouping_option = st.sidebar.selectbox("Select Grouping Option", ['None', 'Position', 'Team'], key="grouping_option")

    if grouping_option == 'None':
        set_index_to_player = st.sidebar.checkbox('Set index to Player', False)

    if grouping_option != 'None':
        grouped_data = group_data(filtered_data, selected_columns, grouping_option, selected_aggregation_method)
        print(f"Grouped Data columns: {grouped_data.columns.tolist()}")
    else:
        grouped_data = filtered_data

        if set_index_to_player:
            grouped_data.set_index('Player', inplace=True)

    grouped_data = grouped_data.applymap(round_and_format)
    columns_to_show = list(DEFAULT_COLUMNS) + selected_columns
    columns_to_show = [col for col in columns_to_show if col in grouped_data.columns]

    if grouping_option != 'None':
        if grouping_option.capitalize() not in columns_to_show:
            columns_to_show.insert(0, grouping_option.capitalize())

    styled_df = style_dataframe_custom(grouped_data[columns_to_show], columns_to_show, False)

    st.header(f"Premier League Players' Statistics ({selected_group})")

    # print columns as list
    print("Grouped Data Columns: ", print(grouped_data[columns_to_show].columns.tolist()))

    print("Styled Data Columns: ", print(styled_df.columns.tolist()))

    st.table(styled_df)

    st.dataframe(grouped_data[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=(len(grouped_data) * 30) + 50 if grouping_option != 'None' else 35 * 20)

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


