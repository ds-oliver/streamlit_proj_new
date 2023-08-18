import streamlit as st
import pandas as pd
import numpy as np
# import sys
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

# logger = st.logger

warnings.filterwarnings('ignore')

# Adding path to the scripts directory
scripts_path = os.path.abspath(os.path.join('./scripts'))

st.set_page_config(
    layout="wide"
)
fbref_cats = ['stats', 'shooting', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'playingtime', 'misc']

fbref_leagues = ['Big5', 'ENG', 'ESP', 'ITA', 'GER', 'FRA']

seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

stats_cols = ['goals', 'assists', 'goals_assists', 'goals_pens', 'pens_made', 'pens_att', 'cards_yellow', 'cards_red', 'xg', 'npxg', 'xg_assist', 'npxg_xg_assist', 'progressive_carries', 'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90', 'goals_assists_per90', 'goals_pens_per90', 'goals_assists_pens_per90', 'xg_per90', 'xg_assist_per90', 'xg_xg_assist_per90', 'npxg_per90', 'npxg_xg_assist_per90']

shooting_cols = ['shots', 'shots_on_target', 'shots_on_target_pct', 'shots_per90', 'shots_on_target_per90', 'goals_per_shot', 'goals_per_shot_on_target', 'average_shot_distance', 'shots_free_kicks', 'pens_made', 'pens_att', 'xg', 'npxg', 'npxg_per_shot', 'xg_net', 'npxg_net']

passing_cols = ['passes_completed', 'passes', 'passes_pct', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'assists', 'xg_assist', 'pass_xa', 'xg_assist_net', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes']

passing_types_cols = ['passes', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_completed', 'passes_offsides', 'passes_blocked']

gca_cols = ['sca', 'sca_per90', 'sca_passes_live', 'sca_passes_dead', 'sca_take_ons', 'sca_shots', 'sca_fouled', 'sca_defense', 'gca', 'gca_per90', 'gca_passes_live', 'gca_passes_dead', 'gca_take_ons', 'gca_shots', 'gca_fouled', 'gca_defense']

defense_cols = ['tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocks', 'blocked_shots', 'blocked_passes', 'interceptions', 'tackles_interceptions', 'clearances', 'errors']

possession_cols = ['touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons', 'take_ons_won', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries', 'carries_distance', 'carries_progressive_distance', 'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received']

playing_time_cols = ['minutes_per_game', 'minutes_pct', 'minutes_90s', 'games_starts', 'minutes_per_start', 'games_complete', 'games_subs', 'minutes_per_sub', 'unused_subs', 'points_per_game', 'on_goals_for', 'on_goals_against', 'plus_minus', 'plus_minus_per90', 'plus_minus_wowy', 'on_xg_for', 'on_xg_against', 'xg_plus_minus', 'xg_plus_minus_per90', 'xg_plus_minus_wowy']

misc_cols = ['cards_yellow', 'cards_red', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'crosses', 'interceptions', 'tackles_won', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

from files import pl_data_gw1, temp_gw1_fantrax_default as temp_default # this is the file we want to read in

from functions import scraping_current_fbref, normalize_encoding, clean_age_column, create_sidebar_multiselect

def style_dataframe(df, selected_columns):
    styled_df = df.copy()

    for col in selected_columns:
        if df[col].dtype in [np.float64, np.int64]:
            min_val = df[col].min()
            range_val = df[col].max() - min_val
            rgba_colors = [f'background-color: rgba({",".join(map(str, ((color[:3] * 255).astype(int))))}, 0.7)' for color in cm.get_cmap('coolwarm')((df[col] - min_val) / range_val)]
            styled_df[col] = rgba_colors
        elif df[col].dtype == 'object':
            unique_values = df[col].unique().tolist()
            object_cmap = cm.get_cmap('viridis')
            styled_df[col] = [get_color(val, unique_values, object_cmap) for val in df[col]]

    return styled_df

def get_color(val, unique_values, object_cmap):
    norm = plt.Normalize(0, len(unique_values) - 1)
    rgba_color = object_cmap(norm(unique_values.index(val)))
    return f'background-color: rgba({int(rgba_color[0] * 255)}, {int(rgba_color[1] * 255)}, {int(rgba_color[2] * 255)}, 0.7)'

# from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols

# Read the data
df = pd.read_csv(pl_data_gw1)
temp_df = pd.read_csv(temp_default)
df['fantrax position'] = temp_df['Position']

# drop df['position'] column
df.drop(columns=['position'], inplace=True)

# Define default columns
DEFAULT_COLUMNS = ['player', 'fantrax position', 'team', 'games_starts']

# Exclude the default columns
stat_cols = [col for col in df.columns if col not in DEFAULT_COLUMNS]

# create a multiselect for the teams, default to all teams
selected_teams = create_sidebar_multiselect(df, 'team', 'Select Teams', default_all=True)

# create a multiselect for the positions, default to all positions
selected_positions = create_sidebar_multiselect(df, 'fantrax position', 'Select Positions', default_all=True)

# Filter the DataFrame for selected teams and positions
df = df[df['team'].isin(selected_teams) & df['fantrax position'].isin(selected_positions)]

# if there are no teams or positions selected, display a message
if len(selected_teams) == 0 or len(selected_positions) == 0:
    st.write('Please select at least one team and one position.')

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

def get_grouped_data(df, group_by, aggregation_func):
    if group_by == 'None':
        return df
    group_column = 'fantrax position' if group_by == 'Position' else 'team'
    return df.groupby(group_column).agg(aggregation_func).reset_index().round(2)

columns_to_show = DEFAULT_COLUMNS + selected_columns
col1, col2 = st.columns(2)

grouping_option = col1.radio('Group Data by:', ('None', 'Position', 'Team'))
aggregation_func = col2.radio('Select Aggregate:', ('Mean', 'Median', 'Sum')).lower() if grouping_option != 'None' else None

grouped_df = get_grouped_data(df, grouping_option, aggregation_func)

styled_df = style_dataframe(grouped_df[columns_to_show], selected_columns)
st.table(styled_df, use_container_width=True)



# Check if there are selected groups and columns
if selected_group and selected_columns:
    selected_stats_for_plot = st.multiselect('Select Statistics for Plotting', options=selected_columns)
    st.info('Note: If no grouping option is selected, the top 25 players by the first selected statistic is shown.')

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
            # get the top 25 players by all of the selected stats
            top_players = grouped_df.nlargest(25, selected_stats_for_plot)
            grouping_values = top_players['player'].tolist()
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






