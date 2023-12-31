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
from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, colors

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default, matches_data, shots_data # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect, create_custom_cmap, style_dataframe_custom, style_tp_dataframe_custom, add_construction, load_css, create_custom_sequential_cmap

st.set_page_config(
    page_title="Footy Magic",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={

    }
)

# scraped data from : /Users/hogan/dev/fbref/scripts/rfx_scrape/match_results_project/premierleague/main/test-weekly-match-reports.py

# logger = st.logger

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

print("Scripts path:", scripts_path)

print(sys.path)

load_css()

# def get_color(value, cmap):
#     color_fraction = value
#     rgba_color = cmap(color_fraction)
#     brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
#     text_color = 'white' if brightness < 0.7 else 'black'
#     return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

# def get_color_from_palette(value, palette_name='inferno'):
#     cmap = cm.get_cmap(palette_name)
#     rgba_color = cmap(value)
#     color_as_hex = mcolors.to_hex(rgba_color)
#     return color_as_hex

# def style_dataframe(df, selected_columns):
#     object_cmap = cm.get_cmap('gnuplot2')

#     # Create an empty DataFrame with the same shape as df
#     styled_df = pd.DataFrame('', index=df.index, columns=df.columns)

#     if 'Pos' in df.columns:
#         unique_positions = df['Pos'].unique().tolist()

#         # Define the colors for the positions
#         position_colors = {
#             "D": "background-color: #3d0b4d;",  # Specific purple color for "D"
#             "M": "background-color: #08040f",  # Assigned color for "M"
#             "F": "background-color: #050255"   # Assigned color for "F"
#         }

#         # Apply the colors to the 'Pos' and 'Player' columns
#         styled_df['Pos'] = df['Pos'].apply(lambda x: position_colors[x])
#         styled_df['Player'] = df['Pos'].apply(lambda x: position_colors[x])

#     for col in df.columns:
#         if col in ['Player', 'Pos']:
#             continue

#         col_dtype = df[col].dtype
#         unique_values = df[col].unique().tolist()

#         if len(unique_values) <= 3:
#             constant_colors = [get_color(i / 2, cm.get_cmap('inferno')) for i in range(len(unique_values))]
#             color_mapping = {val: color for val, color in zip(unique_values, constant_colors)}
#             styled_df[col] = df[col].apply(lambda x: color_mapping[x])
#         elif col_dtype in [np.float64, np.int64] and col in selected_columns:
#             min_val = df[col].min()
#             max_val = df[col].max()
#             range_val = max_val - min_val
#             styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / range_val, cm.get_cmap('inferno')))
#         elif col_dtype == 'object':
#             styled_df[col] = df[col].apply(lambda x: get_color(unique_values.index(x) / len(unique_values), object_cmap))

#     return styled_df

@st.cache_data
def read_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def process_matches_data(matches_data, temp_data, matches_drop_cols):
    matches_df = read_data(matches_data)
    temp_df = read_data(temp_data)
    print("Shape of matches_df before merging:", matches_df.shape)

    # drop where position_1 is 'GK'
    matches_df = matches_df[matches_df['position_1'] != 'GK']

    matches_df = pd.merge(matches_df, temp_df[['Player', 'Position', 'Team']], left_on='player', right_on='Player', how='left')
    print("Shape of matches_df after merging:", matches_df.shape)

    # drop the 'Player' and 'team' columns
    matches_df.drop(columns=['Player', 'team'], inplace=True)

    matches_df.rename(columns={'Position': 'Pos'}, inplace=True)

    # drop rows where Pos is nan
    matches_df = matches_df[matches_df['Pos'].notna()]

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

    # rename any column name that starts with xg to xG
    matches_df.rename(columns={col: col.replace('X', 'x') for col in matches_df.columns if col.startswith('X')}, inplace=True)

    # if the column name starts with x and is 2 letters in length capitalize the second letter, if it starts with x and is 3 letters in length capitalize the second and third letters
    matches_df.rename(columns={col: col[:2] + col[2:].capitalize() if col.startswith('x') and len(col) == 2 else col[:2] + col[2:4].capitalize() + col[4:] if col.startswith('x') and len(col) == 3 else col for col in matches_df.columns if col.startswith('x')}, inplace=True)

    print("Columns in matches_df after capitalizing:", matches_df.columns.tolist())

    MATCHES_DEFAULT_COLS = matches_default_cols

    # capitalize format for the columns
    MATCHES_DEFAULT_COLS = [col.capitalize() if col != 'GW' else col for col in MATCHES_DEFAULT_COLS]

    print("Default columns:", MATCHES_DEFAULT_COLS)
    
    return matches_df, MATCHES_DEFAULT_COLS


@st.cache_data
def load_shots_data(shots_data):
    return read_data(shots_data)

def create_top_performers_table(matches_df, selected_stat, selected_columns, percentile=85):
    top_performers_df = matches_df.copy()
    cols_to_drop = [col for col in top_performers_df.columns if col not in selected_columns + ['player', 'team', 'position']]
    top_performers_df.drop(columns=cols_to_drop, inplace=True)

    # Calculate the 90th percentile for the selected stat
    threshold_value = top_performers_df[selected_stat].quantile(percentile / 100)

    # Filter the dataframe by the 90th percentile
    top_performers_df = top_performers_df[top_performers_df[selected_stat] >= threshold_value]

    # Only return players where stat is not 0
    top_performers_df = top_performers_df[top_performers_df[selected_stat] > 0]
    
    # round the selected stat to 2 decimal places
    top_performers_df[selected_stat] = top_performers_df[selected_stat].round(2)

    # Sort the dataframe by the selected stat in descending order and take the top 25
    top_performers_df = top_performers_df.sort_values(by=selected_stat, ascending=False)

    # reset the index
    top_performers_df.reset_index(drop=True, inplace=True)

    return top_performers_df

def visualize_top_performers(top_performers_df, selected_stat):
    # plot the top performers just the players and the selected stat
    fig = px.bar(top_performers_df, x='player', y=selected_stat, color='team', color_discrete_sequence=px.colors.qualitative.Pastel)

    # update the layout
    fig.update_layout(
        title=f'Top performers for {selected_stat}',
        xaxis_title='Player',
        yaxis_title=selected_stat,
        legend_title='Team',
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        )
    )

    # display the plot
    st.plotly_chart(fig)

@st.cache_data
def process_data():
    matches_df, MATCHES_DEFAULT_COLS = process_matches_data(matches_data, temp_default, matches_drop_cols)
    shots_df = load_shots_data(shots_data)
    date_of_update = datetime.fromtimestamp(os.path.getmtime(matches_data)).strftime('%d %B %Y')
    return matches_df, shots_df, date_of_update, MATCHES_DEFAULT_COLS

def display_date_of_update(date_of_update):
    st.write(f'Last data refresh: {date_of_update}')

def round_and_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value

def create_top_teams_table(teams_df, selected_stat):
    top_teams_df = teams_df.copy()
    cols_to_drop = [col for col in top_teams_df.columns if col not in ['Team', 'GP', 'Player Count', 'Subs Used']]
    top_teams_df.drop(columns=cols_to_drop, inplace=True)

    # Sort the dataframe by the selected stat in descending order
    top_teams_df = top_teams_df.sort_values(by=selected_stat, ascending=False)

    # reset the index
    top_teams_df.reset_index(drop=True, inplace=True)

    return top_teams_df

# Assuming 'teams_df' is your grouped dataframe and 'original_df' contains raw data
def re_add_aggregated_column(teams_df, original_df, selected_stat):
    aggregated_stat = original_df.groupby('Team')[selected_stat].sum().reset_index()
    teams_df = pd.merge(teams_df, aggregated_stat, on='Team', how='left')
    return teams_df


def main():

    add_construction()

    custom_cmap = create_custom_sequential_cmap(*colors)

    # Load the data
    matches_df, shots_df, date_of_update, MATCHES_DEFAULT_COLS = process_data()

    # reorder the columns in MATCHES_DEFAULT_COLS so that its ['Player', 'Pos', 'Team', 'GW', 'Started']
    MATCHES_DEFAULT_COLS = ['Player', 'Pos', 'Team', 'GW', 'Started'] + [col for col in MATCHES_DEFAULT_COLS if col not in ['Player', 'Pos', 'Team', 'GW', 'Started']]

    TEAMS_DEFAULT_COLUMNS = ['Team', 'GP', 'Player Count', 'Subs Used']

    # print MATCHES_DEFAULT_COLS
    print("MATCHES_DEFAULT_COLS:", MATCHES_DEFAULT_COLS)

    matches_col_groups = {
    "Standard": matches_standard_cols,
    "Passing": matches_passing_cols,
    "Defense": matches_defense_cols,
    "Possession": matches_possession_cols,
    # "Miscellaneous": matches_misc_cols,
    "Passing Types": matches_pass_types
}
    matches_col_groups = {key.capitalize(): [col.capitalize() for col in value] for key, value in matches_col_groups.items()}
    # capitalize the contents of the lists in matches_col_groups dictionary

    # custom_cmap = create_custom_cmap('#e63946', '#f1faee','#a8dadc', '#457b9d', '#e63946')

    print(matches_df.columns.tolist())

    display_date_of_update(date_of_update)

    # state at the top of the page as header the grouping option selected
    st.header(f"Premier League Individual Players' Data")

    all_stats = matches_df.columns.tolist()
    selectable_stats = [stat for stat in all_stats if stat not in MATCHES_DEFAULT_COLS]

    default_stat_index = 13

    # User selects the stat
    selected_stat = st.sidebar.selectbox("Select a Statistic for **Top Performers**", selectable_stats, index=default_stat_index, help="This selection will be used to populate the Top Performers table.", key='stat')

    # User selects the group and columns to show
    selected_group = st.sidebar.selectbox("Select Stats Grouping", list(matches_col_groups.keys()), help="This selection will be used to populate the Match Reports table.", key='group')

    # create radio button for 'Starting XI' or 'All Featured Players'
    featured_players = st.sidebar.radio("Select Featured Players", ('All Featured Players', 'Starting XI', '> 55 Minutes Played'))

    selected_position = st.sidebar.selectbox('Select Position', ['All Positions'] + sorted(matches_df['Pos'].unique().tolist()))

    selected_team = st.sidebar.selectbox('Select Team', ['All Teams'] + sorted(matches_df['Team'].unique().tolist()))

    if selected_position != 'All Positions':
        matches_df = matches_df[matches_df['Pos'] == selected_position]

    if selected_team != 'All Teams':
        matches_df = matches_df[matches_df['Team'] == selected_team]

    # # print count of featured players
    # st.sidebar.write(f'Number of featured players: {matches_df.shape[0]}')

    # # print count of starting XI
    # st.sidebar.write(f'Number of starting XI: {matches_df[matches_df["started"] > 0].shape[0]}')

    # If the column 'Gw' exists, rename it to 'GW'
    if 'Gw' in matches_df.columns:
        matches_df.rename(columns={'Gw': 'GW'}, inplace=True)

    # Filter the DataFrame based on the radio button selected, set it to All Featured Players by default
    if featured_players == 'All Featured Players':
        matches_df = matches_df

    if featured_players == '> 55 Minutes Played':
        matches_df = matches_df[matches_df['Minutes'] > 55]

    # Create a sidebar slider to select the GW range
    GW_range = st.sidebar.slider('GW range', 
                             min_value=int(matches_df['GW'].min()), 
                             max_value=int(matches_df['GW'].max()), 
                             value=(int(matches_df['GW'].min()), int(matches_df['GW'].max())), 
                             step=1, 
                             help="Select the range of gameweeks to display data for. This slider adjusts data globally for all tables and plots")

    GW_range = list(GW_range)

    # Filter the DataFrame by the selected GW range
    matches_df = matches_df[(matches_df['GW'] >= GW_range[0]) & (matches_df['GW'] <= GW_range[1])]
    print("Shape of matches_df after filtering by featured players:", matches_df.shape)

    # Allow the user to select the aggregation method
    selected_aggregation_method = st.sidebar.selectbox('Select Aggregation Method', ['mean', 'sum'])

    # If the GW_range list has more than 1 element, group by MATCHES_DEFAULT_COLS
    if GW_range[0] != GW_range[1]:

        # Define aggregation functions for numeric and non-numeric columns
        aggregation_functions = {col: selected_aggregation_method if matches_df[col].dtype in [np.float64, np.int64] else 'first' for col in matches_df.columns}
        aggregation_functions['Player'] = 'first'
        aggregation_functions['Team'] = 'first'
        aggregation_functions['Pos'] = 'first' # Aggregating by the first occurrence of position
        aggregation_functions['GW'] = 'nunique' # Counting the number of GWs
        aggregation_functions['Started'] = 'sum' # Summing the number of starts

        # Group by player, team, and position, and apply the aggregation functions
        matches_df = matches_df.groupby(['Player', 'Team', 'Pos'], as_index=False).agg(aggregation_functions)
        print("Shape of matches_df after grouping by player, team, and position:", matches_df.shape)

        # Rename the 'GW' column to 'GP' (games played) and 'Started' column to 'GS' (games started)
        matches_df.rename(columns={'GW': 'GP', 'Started': 'GS'}, inplace=True)

        # Update MATCHES_DEFAULT_COLS
        MATCHES_DEFAULT_COLS = [col if col != 'GW' else 'GP' for col in MATCHES_DEFAULT_COLS]
        MATCHES_DEFAULT_COLS = [col if col != 'Started' else 'GS' for col in MATCHES_DEFAULT_COLS]

        # Create a ratio of potential GP to GS and append the ratio to MATCHES_DEFAULT_COLS
        matches_df['GS:GP'] = round(matches_df['GS'] / matches_df['GP'].max(), 2).apply(lambda x: f"{x:.2f}")
        MATCHES_DEFAULT_COLS.append('GS:GP')

        # Remove GP from MATCHES_DEFAULT_COLS and reorder the columns
        MATCHES_DEFAULT_COLS.remove('GP')
        MATCHES_DEFAULT_COLS = ['Player', 'Pos', 'Team', 'GS:GP'] + [col for col in MATCHES_DEFAULT_COLS if col not in ['Player', 'Team', 'Pos', 'GS:GP']]

        # Now we create the dataframe grouped by Team before any filtering is applied, for teams_df we want Count of player
        teams_df = matches_df.groupby(['Team'], as_index=False).agg({'Player': 'count', 'GP': 'first'})
        teams_df.rename(columns={'Player': 'Player Count'}, inplace=True)

        # Create Subs_Used column that takes count of players minus 11
        teams_df['Subs Used'] = teams_df['Player Count'] - 11

        teams_df = re_add_aggregated_column(teams_df, matches_df, selected_stat)

        # If the user selects 'Starting XI', filter the DataFrame accordingly
        if featured_players == 'Starting XI':
            matches_df['GS:GP'] = matches_df['GS:GP'].astype(float).round(2)  # Convert the 'GS:GP' column to float
            matches_df = matches_df[(matches_df['GS'] > 0) & (matches_df['GS:GP'] >= 0.50)]

            # Create an info message to explain the ratio as the percentage of games started out of total possible games
            st.write(f'You have chosen to retrieve data from **:red[GW {GW_range[0]}]** to **:red[GW {GW_range[1]}]** for **:red[{selected_position}]**. \nAs you have selected multiple GWs, and filtered for "Starting XI" players, **:orange[only]** players that have a GS to GP ratio of at least **:orange[0.50]** (i.e. **players who have started a minimum of half of all possible games**) will be shown.')
        else:
            st.write(f'You have chosen to retrieve data from **:red[GW {GW_range[0]}]** to **:red[GW {GW_range[1]}]** for **:orange[{selected_position}]**. The data is aggregated by **:orange[{selected_aggregation_method}]**.')
        

    else:
        matches_df.rename(columns={'GW': 'GP', 'Started': 'GS'}, inplace=True)
        # Show a message for the selected GW
        st.info(f'GW {GW_range[0]} selected')

    if featured_players == 'Starting XI':
        matches_df = matches_df[(matches_df['GS'] > 0)]

    # all_stats = matches_df.columns.tolist()
    # selectable_stats = [stat for stat in all_stats if stat not in MATCHES_DEFAULT_COLS]

    # default_stat_index = 13

    # # User selects the stat
    # selected_stat = st.sidebar.selectbox("Select a Statistic", selectable_stats, index=default_stat_index, help="This selection will be used to populate the Top Performers table.")
    columns_to_show = MATCHES_DEFAULT_COLS + [selected_stat]

    teams_columns_to_show = TEAMS_DEFAULT_COLUMNS + [selected_stat]

    # Create a DataFrame for players within the 90th percentile
    top_performers_df = create_top_performers_table(matches_df, selected_stat, columns_to_show)
    # if team != 'All Teams': or position != 'All Positions': or both then change the percentile to 50th
    percentile_value = 85
    if selected_team != 'All Teams' or selected_position != 'All Positions':
        percentile_value = 50
        top_performers_df = create_top_performers_table(matches_df, selected_stat, columns_to_show, percentile=percentile_value)

    st.divider()
    st.subheader(f":orange[Top Performers] by :red[{selected_stat} ({selected_aggregation_method})]")
    st.write(f'**{selected_team}** players in the {percentile_value}th percentile by **:red[{selected_stat} ({selected_aggregation_method})]. Conditional formatting has been applied to Team value based on rank of {selected_stat} ({selected_aggregation_method})**')

    # Styling DataFrame
    styled_top_performers_df = style_dataframe_custom(top_performers_df[columns_to_show], columns_to_show, custom_cmap=custom_cmap)

    top_performers_df[selected_stat] = top_performers_df[selected_stat].apply(lambda x: f"{x:.2f}")
    top_performers_df = top_performers_df.applymap(round_and_format)

    # 👈 Display the dataframe 👈
    st.dataframe(top_performers_df[columns_to_show].style.apply(lambda _: styled_top_performers_df, axis=None), use_container_width=True, height=len(top_performers_df) * 40 + 50)

    st.divider()

    # Now we show the teams_df dataframe
    st.subheader(f":orange[Teams]")

    top_teams_df = create_top_teams_table(teams_df, selected_stat)

    styled_teams_df = style_dataframe_custom(top_teams_df[teams_columns_to_show], teams_columns_to_show, custom_cmap=custom_cmap)
    st.dataframe(teams_df.style.apply(lambda _: styled_teams_df, axis=None), use_container_width=True, height=len(teams_df) * 40 + 50)

    # Create dataframe grouped by team
    st.divider()

    # User selects the group and columns to show
    # selected_group = st.sidebar.selectbox("Select Stats Grouping", list(matches_col_groups.keys()), help="This selection will be used to populate the Match Reports table.")
    selected_columns = matches_col_groups[selected_group]
    # matches_df[selected_group] = matches_df[selected_group].apply(lambda x: f"{x:.2f}")
    columns_to_show = MATCHES_DEFAULT_COLS + [col for col in selected_columns if col in matches_df.columns]

    matches_df = matches_df.applymap(round_and_format)

    st.subheader(f":orange[Match Reports] by :red[{selected_group}] grouping")

    st.info(f'**:green[{matches_df.shape[0]}]** players found within the parameters selected', icon='ℹ')

    # Styling DataFrame
    styled_df = style_dataframe_custom(matches_df[columns_to_show], columns_to_show, custom_cmap=custom_cmap)

    # display the dataframe
    # 👈 round the values of the columns 👈
    matches_df = matches_df.round(2)
    st.dataframe(matches_df[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(matches_df) * 40)


if __name__ == "__main__":
    main()