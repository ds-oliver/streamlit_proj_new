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

scripts_path = os.path.abspath(os.path.join('..', 'scripts'))
sys.path.append(scripts_path)
print("Scripts path:", scripts_path)

print(sys.path)

st.set_page_config(
    layout="wide"
)

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues


# fbref_cats = ['stats', 'shooting', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'playingtime', 'misc']

# fbref_leagues = ['Big5', 'ENG', 'ESP', 'ITA', 'GER', 'FRA']

# seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

# stats_cols = ['goals', 'assists', 'goals_assists', 'goals_pens', 'pens_made', 'pens_att', 'cards_yellow', 'cards_red', 'xg', 'npxg', 'xg_assist', 'npxg_xg_assist', 'progressive_carries', 'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90', 'goals_assists_per90', 'goals_pens_per90', 'goals_assists_pens_per90', 'xg_per90', 'xg_assist_per90', 'xg_xg_assist_per90', 'npxg_per90', 'npxg_xg_assist_per90']

# shooting_cols = ['shots', 'shots_on_target', 'shots_on_target_pct', 'shots_per90', 'shots_on_target_per90', 'goals_per_shot', 'goals_per_shot_on_target', 'average_shot_distance', 'shots_free_kicks', 'pens_made', 'pens_att', 'xg', 'npxg', 'npxg_per_shot', 'xg_net', 'npxg_net']

# passing_cols = ['passes_completed', 'passes', 'passes_pct', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'assists', 'xg_assist', 'pass_xa', 'xg_assist_net', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes']

# passing_types_cols = ['passes', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_completed', 'passes_offsides', 'passes_blocked']

# gca_cols = ['sca', 'sca_per90', 'sca_passes_live', 'sca_passes_dead', 'sca_take_ons', 'sca_shots', 'sca_fouled', 'sca_defense', 'gca', 'gca_per90', 'gca_passes_live', 'gca_passes_dead', 'gca_take_ons', 'gca_shots', 'gca_fouled', 'gca_defense']

# defense_cols = ['tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocks', 'blocked_shots', 'blocked_passes', 'interceptions', 'tackles_interceptions', 'clearances', 'errors']

# possession_cols = ['touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons', 'take_ons_won', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries', 'carries_distance', 'carries_progressive_distance', 'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received']

# playing_time_cols = ['minutes_per_game', 'minutes_pct', 'minutes_90s', 'games_starts', 'minutes_per_start', 'games_complete', 'games_subs', 'minutes_per_sub', 'unused_subs', 'points_per_game', 'on_goals_for', 'on_goals_against', 'plus_minus', 'plus_minus_per90', 'plus_minus_wowy', 'on_xg_for', 'on_xg_against', 'xg_plus_minus', 'xg_plus_minus_per90', 'xg_plus_minus_wowy']

# misc_cols = ['cards_yellow', 'cards_red', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'crosses', 'interceptions', 'tackles_won', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

def get_color(value, unique_values, cmap):
    index = unique_values.index(value)
    color_fraction = index / len(unique_values)
    rgba_color = cmap(color_fraction)
    return f'background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

def style_dataframe(df, selected_columns):
    cm_coolwarm = cm.get_cmap('coolwarm')
    object_cmap = cm.get_cmap('viridis')  # Choose a colormap for object columns

    # Create an empty DataFrame with the same shape as df
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    for col in df.columns:
        if col == 'player':  # Skip the styling for the 'player' column
            continue
        if df[col].dtype in [np.float64, np.int64] and col in selected_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            styled_df[col] = df[col].apply(lambda x: f'background-color: rgba({",".join(map(str, (np.array(cm_coolwarm((x - min_val) / range_val))[:3] * 255).astype(int)))}, 0.7)')
        elif df[col].dtype == 'object':
            unique_values = df[col].unique().tolist()
            styled_df[col] = df[col].apply(lambda x: get_color(x, unique_values, object_cmap))
    return styled_df

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

    return df, DEFAULT_COLUMNS

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
    data, DEFAULT_COLUMNS = load_data()

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
    styled_df = style_dataframe(grouped_data[columns_to_show], selected_columns=selected_columns)

    # Display the DataFrame
    # if grouping_option == 'None' then set st.dataframe() height= to the height of showing the first 50 rows, else set height to length of grouped_data
    if grouping_option == 'None':
        st.dataframe(grouped_data[columns_to_show].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=50 * 20)
    else:
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


