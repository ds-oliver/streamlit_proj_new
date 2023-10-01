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
import plost
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import warnings
import requests
from bs4 import BeautifulSoup
import unidecode
import matplotlib.cm as mpl_cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import get_cmap
import matplotlib
from collections import Counter
from markdownlit import mdlit
from scipy.stats import percentileofscore


df = pd.DataFrame()
df2 = pd.DataFrame()

sys.path.append(os.path.abspath(os.path.join('./scripts')))

from constants import color1, color2, color3, color4, color5, cm

def add_construction():
    return st.info("""🏗️ **:orange[This app is under construction]**""")

def load_css(file_name="style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def display_date_of_update(date_of_update):
    st.write(f'Last data refresh: {date_of_update}')

def load_data_from_db():
    """
    This function loads data from two tables in a SQLite database.
    Returns two pandas DataFrames, one for each table.
    """
    # Create a connection to the database
    conn = sqlite3.connect(
        'data/data_out/final_data/db_files/results.db')

    # Load data from the 'players' table into a DataFrame
    players_table = pd.read_sql_query("SELECT * from players", conn)

    # Load data from the 'results' table into a DataFrame
    results_table = pd.read_sql_query("SELECT * from results", conn)

    # Don't forget to close the connection when you're done
    conn.close()

    return players_table, results_table

def load_data_from_csv():
    """
    This function loads the data from the csvs.
    """
    # Load the data from the csv

    players_df = pd.read_csv(
        "data/data_out/final_data/csv_files/players.csv")
    results_df = pd.read_csv(   
        "data/data_out/final_data/csv_files/results.csv")
    
    return players_df, results_df

def load_data_from_pkl():
    """
    This function loads the data from the pkl.
    """
    # Load the data from the csv

    master_dict = pd.read_pickle(
        "data/data_out/final_data/pickle_files/master_dict.pickle")
    results_dict = pd.read_pickle(   
        "data/data_out/final_data/pickle_files/only_results_dict.pkl")
    players_dict = pd.read_pickle(
        'data/data_out/final_data/pickle_files/players_results_dict.pkl')
    
    return players_dict, results_dict, master_dict

def select_season_from_db(db, season):
    """
    This function selects the season from the db.
    """
    # Select the season from the db using sqlite3
    cur = db.cursor()
    cur.execute(
        "SELECT * FROM results WHERE season = ?", (season,))
    results = cur.fetchall()

    return results

def print_df_columns(df):
    """_summary_

    Args:
        df (_type_): _description_
    """
    print(f"<> Printing df columns:\n   ---------------------------------------------------------------\n   == {df.columns.tolist()}")

def color_format(results):
    formatted_results = []
    color_map = {"W": color1, "L": color2, "D": color3}

    for result in results:
        color = color_map[result]
        formatted_result = f'<span style="color:{color};">{result}</span>'
        formatted_results.append(formatted_result)

    return ", ".join(formatted_results)

def show_dataframe(df):
    """_summary_

    Args:
        teams_stats (_type_): _description_
    """
    st.dataframe(df, use_container_width=True)

def show_teams_stats_v2(team_stats_df):

    # Qualitative statistics dataframe
    qual_stats_df = team_stats_df.loc[['Total Games', 'Total Wins', 'Total Losses'], :]

    st.data_editor(
        qual_stats_df,
        column_config={
            qual_stats_df.columns[0]: st.column_config.ListColumn(qual_stats_df.columns[0]),
            qual_stats_df.columns[1]: st.column_config.ListColumn(qual_stats_df.columns[1]),
        },
        hide_index=True,
    )

    # Quantitative statistics dataframe
    quant_stats_df = team_stats_df.loc[['Total Goals Scored', 'Total Goals Conceded', 'xG For', 'xG Against', 'Clean Sheets'], :]

    st.data_editor(
        quant_stats_df,
        column_config={
            quant_stats_df.columns[0]: st.column_config.BarChartColumn(quant_stats_df.columns[0], y_min=0),
            quant_stats_df.columns[1]: st.column_config.BarChartColumn(quant_stats_df.columns[1], y_min=0),
        },
        hide_index=True,
    )

def get_results_list(selected_teams_df, team):
    home_condition = (selected_teams_df['home_team'] == team)
    away_condition = (selected_teams_df['away_team'] == team)
    
    conditions = [
        (home_condition & (selected_teams_df['home_score'] > selected_teams_df['away_score'])) | (away_condition & (selected_teams_df['home_score'] < selected_teams_df['away_score'])),
        (home_condition & (selected_teams_df['home_score'] < selected_teams_df['away_score'])) | (away_condition & (selected_teams_df['home_score'] > selected_teams_df['away_score'])),
        (selected_teams_df['home_score'] == selected_teams_df['away_score'])
    ]
    choices = ['W', 'L', 'D']
    
    team_results = np.select(conditions, choices, default=np.nan)
    
    return team_results.tolist()

def get_results_df(selected_teams_df, selected_team, selected_opponent):
    # Filter for matches involving the selected teams
    condition = (selected_teams_df['home_team'].isin([selected_team, selected_opponent])) & (selected_teams_df['away_team'].isin([selected_team, selected_opponent]))
    results_df = selected_teams_df[condition].copy()
    
    return results_df

def match_quick_facts(selected_teams_df, player_level_df, selected_team, selected_opponent):
    """Summary:
        This function returns the quick facts for the selected team and opponent using fstrings and vectorized computations.
        These stats are:
        - Total Games
        - Total Goals
        - Total players used
        - Top player by appearances
        - Teams' average attendance

    Args:
        df (pandas DataFrame): The df to be filtered.
        selected_team (str): The selected team.
        selected_opponent (str): The selected opponent.
    """
    
    # st.dataframe(player_level_df, use_container_width=True)
    st.info("Matchup quick facts below...")

    # Get team and opponent stats
    team_stats_og, _ = get_teams_stats(selected_teams_df, selected_team, selected_opponent)

    # Get player stats
    for team in [selected_team, selected_opponent]:
        # Get the number of players used by the team
        total_players = player_level_df[player_level_df['team'] == team]['player'].nunique()

        # Get the top player by appearances
        top_player = player_level_df[player_level_df['team'] == team].groupby('player')['player'].count().idxmax()

        # Print the quick facts
        st.metric(label=f"{team} - Total Players Used", value=total_players)
        st.metric(label=f"{team} - Top Player by Appearances", value=top_player)

    # Show team level stats with st.metric
    for stat in team_stats_og.columns:
        delta = team_stats_og.loc[selected_team, stat] - team_stats_og.loc[selected_opponent, stat]
        delta_color = "normal" # if delta > 0 else "inverse"
        winner = selected_team if delta > 0 else selected_opponent
        value = team_stats_og.loc[winner, stat]
        st.metric(label=f"Most {stat} - {winner}", value=value, delta=f"{abs(delta)} more than {selected_opponent if winner == selected_team else selected_team}", delta_color=delta_color)

def style_dataframe(df, cm):
    # Create two diverging palettes
    cm = sns.diverging_palette(255, 149, s=80, l=55, as_cmap=True)  # gold palette
    cm2 = sns.diverging_palette(175, 35, s=60, l=85, as_cmap=True)  # light blue palette

    # Apply the color gradient using the two diverging palettes
    styled_df = df.style.background_gradient(cmap=cm, low=df.min().min(), high=0)\
                        .background_gradient(cmap=cm2, low=0, high=df.max().max())
    return styled_df

def display_styled_dataframe(df, subset1, subset2):
    cm = sns.diverging_palette(255, 149, s=80, l=55, as_cmap=True)  # gold palette
    cm2 = sns.diverging_palette(175, 35, s=60, l=85, as_cmap=True)  # light blue palette

    styled_df = df.style.background_gradient(cmap=cm, subset=[subset1], vmin=df[subset1].min(), vmax=df[subset1].max())\
                        .background_gradient(cmap=cm2, subset=[subset2], vmin=df[subset2].min(), vmax=df[subset2].max())
    st.dataframe(styled_df, use_container_width=True)

def display_styled_dataframe_simple(df, cm):

    # Apply color gradients
    styled_df = df.style.background_gradient(cmap=cm)
    
    # Display the styled dataframe
    st.dataframe(styled_df, use_container_width=True)

def display_styled_dataframe_v2(df):
    # Create a color palette that goes from blood red to gold
    cm = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.3, light=0.8, reverse=True, as_cmap=True)

    # Apply color gradients
    styled_df = df.style.background_gradient(cmap=cm)
    
    # Display the styled dataframe
    st.dataframe(styled_df, use_container_width=True)

@st.cache_resource
def clean_data(players_df, results_df):
    """
    This function cleans the data.
    """
    print("Cleaning data...")

    cols_x = {col for col in players_df.columns if col.endswith('_x')}
    cols_y = {col for col in players_df.columns if col.endswith('_y')}
    
    # players_df
    for col in cols_x:
        if col[:-2] not in players_df.columns:
            players_df.rename(columns={col: col[:-2]}, inplace=True)
        else:
            players_df.drop(columns=[col], inplace=True)

    # drop _y columns
    for col in cols_y:
        players_df.drop(columns=[col], inplace=True)
    
    cols_x = {col for col in players_df.columns if col.endswith('_x')}
    cols_y = {col for col in players_df.columns if col.endswith('_y')}
    
    if len(cols_x) > 0:
        print(f"cols_x: {cols_x}")
    else:
        print("No cols with _x found.")
    
    if len(cols_y) > 0:
        print(f"cols_y: {cols_y}")
    else:
        print("No cols with _y found.")

    if 'Unnamed: 0' in players_df.columns:
        print("Unnamed: 0 found")
        players_df.drop(columns=['Unnamed: 0'], inplace=True)

    # change _x columns to columns without _x unless a column without _x already exists and drop _y columns
    for col in cols_x:
        if col[:-2] not in results_df.columns:
            results_df.rename(columns={col: col[:-2]}, inplace=True)
        else:
            results_df.drop(columns=[col], inplace=True)

    # drop _y columns
    for col in cols_y:
        results_df.drop(columns=[col], inplace=True)

    cols_x = {col for col in results_df.columns if col.endswith('_x')}
    cols_y = {col for col in results_df.columns if col.endswith('_y')}

    if len(cols_x) > 0:
        print(f"cols_x: {cols_x}")
    
    if len(cols_y) > 0:
        print(f"cols_y: {cols_y}")

    print(f"Cleaned results_df columns:\n{results_df.columns.tolist()} ")

    # create opponent column that looks at home_team and away_team and assigns the other team to the opponent column

    # create season_matchup_gameweek column that concatenates season, gameweek, home_team, and away_team but it should always be the same order so needs sorted so just add gameweek to season_match_teams column

    players_df['matchup_id'] = players_df['match_teams'].apply(lambda x: '_'.join(sorted(x.split('_'))))

    players_df['season_matchup_id'] = players_df['matchup_id'] + '_' + players_df['season'].astype(str) + '_' + players_df['gameweek'].astype(str)

    # create a new column in the players_df called "opponent"
    team_is_home = players_df["team"] == players_df["home_team"]
    players_df["team_score"] = np.where(team_is_home, players_df["home_score"], players_df["away_score"])
    players_df["opponent_score"] = np.where(team_is_home, players_df["away_score"], players_df["home_score"])

    # convert season to int
    results_df["season"] = results_df["season"].astype(int)
    players_df["season"] = players_df["season"].astype(int)

    # Create a new column in the results_df called "winning_xG"
    players_df["team_xG"] = np.where(team_is_home, players_df["home_xg"], players_df["away_xg"])

    # Create a new column in the results_df called "losing_xG"
    players_df["opponent_xG"] = np.where(team_is_home, players_df["away_xg"], players_df["home_xg"])

    # create team_won column
    players_df["team_won"] = np.where(players_df["team_score"] > players_df["opponent_score"], 1, 0)

    # create team draw column
    players_df["team_draw"] = np.where(players_df["team_score"] == players_df["opponent_score"], 1, 0)

    #create team result column W for win, D for draw, L for loss
    players_df["team_result"] = np.where(players_df["team_score"] > players_df["opponent_score"], "W", np.where(players_df["team_score"] == players_df["opponent_score"], "D", "L"))

    # create opponent_won column
    players_df["opponent_won"] = np.where(players_df["team_score"] < players_df["opponent_score"], 1, 0)

    # create opponent draw column
    players_df["opponent_draw"] = np.where(players_df["team_score"] == players_df["opponent_score"], 1, 0)

    #create opponent result column W for win, D for draw, L for loss
    players_df["opponent_result"] = np.where(players_df["team_score"] < players_df["opponent_score"], "W", np.where(players_df["team_score"] == players_df["opponent_score"], "D", "L"))

    # create team result binary column 1 for win, 0 for draw, -1 for loss
    players_df["team_result_binary"] = np.where(players_df["team_score"] > players_df["opponent_score"], 1, np.where(players_df["team_score"] == players_df["opponent_score"], 0, -1))

    # create opponent result binary column 1 for win, 0 for draw, -1 for loss
    players_df["opponent_result_binary"] = np.where(players_df["team_score"] < players_df["opponent_score"], 1, np.where(players_df["team_score"] == players_df["opponent_score"], 0, -1))

    # create team result str column W, D or L based on team_result_binary 1 is W, 0 is D, -1 is L
    players_df["team_result_str"] = np.where(players_df["team_result_binary"] == 1, "W", np.where(players_df["team_result_binary"] == 0, "D", "L"))

    # create opponent result str column W, D or L based on opponent_result_binary 1 is W, 0 is D, -1 is L
    players_df["opponent_result_str"] = np.where(players_df["opponent_result_binary"] == 1, "W", np.where(players_df["opponent_result_binary"] == 0, "D", "L"))

    # replace value in winning_team and losing_team if = draw with None
    players_df["winning_team"] = np.where(players_df["team_score"] > players_df["opponent_score"], players_df["team"], "None")

    # replace value in winning_team and losing_team if = draw with None
    players_df["losing_team"] = np.where(players_df["winning_team"] == "None", "None", players_df["opponent"])

    # strip special characters from attendance column
    players_df["attendance"] = players_df["attendance"].str.replace(",", "")

    # print this column
    print(f"Printing cleaned players_df:\n{players_df[['player', 'team', 'opponent', 'team_score', 'opponent_score', 'winning_team', 'losing_team', 'home_score', 'away_score', 'team_result', 'team_result_binary', 'team_result_str', 'opponent_result', 'opponent_result_binary', 'opponent_result_str']].head(10)}")

    players_df_copy = players_df[['player', 'team', 'home_team', 'away_team', 'home', 'opponent', 'team_score', 'opponent_score', 'winning_team', 'losing_team', 'home_score', 'away_score', 'team_result', 'team_result_binary', 'team_result_str', 'opponent_result', 'opponent_result_binary', 'opponent_result_str']].head(100)

    # st.info("Displaying top 100 rows of cleaned players_df:")

    # players_df_copy

    # results_df
    print(f"Cleaned players_df columns:\n{players_df.columns.tolist()} ")

    # strip leading or trailing whitespace from all  columns
    players_df = players_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return players_df, results_df

# def process_player_data(players_df):
#     """Separate the columns for player level analysis, we will then create new dataframes and dictionaries for different aggregations of the player data (percent of team total, percent of game total, percent of season, etc.), primarily using per90s"""

#     players_only_df = players_df.copy()

#     players_only_df.sort_values(by=['player', 'season', 'date'], ascending=True)

#     # turn date to numerical value
#     players_only_df['datetime'] = pd.to_datetime(players_only_df['date'])

#     per90_columns = ['minutes', 'goals', 'assists', 'pens_made', 'pens_att', 'shots', 'shots_on_target', 'cards_yellow', 'cards_red', 'touches', 'tackles', 'interceptions', 'blocks', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'passes_completed', 'passes', 'passes_pct', 'progressive_passes', 'carries', 'progressive_carries', 'take_ons', 'take_ons_won', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocked_shots', 'blocked_passes', 'tackles_interceptions', 'clearances', 'errors', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries_distance', 'carries_progressive_distance', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

#     # create per90 minutes columns for all per90 columns
#     for column in per90_columns:
#         players_only_df[f"{column}_per90"] = players_only_df[column] / (players_only_df['minutes'] / 90)
#         # round to 2 decimal places
#         players_only_df[f"{column}_per90"] = players_only_df[f"{column}_per90"].round(2)

#     # rename the columns so that they are more descriptive
#     rename_dict = {
#         f"{column}_per90": " ".join(
#             word.title() if not word.startswith('x') else 'x' + word[1:].capitalize() 
#             for word in column.replace('_', ' ').split()
#         ) + " Per90" for column in per90_columns
#     }
#     players_only_df.rename(columns=rename_dict, inplace=True)

#     # now we do the same formatting for other columns where we replace underscores with spaces and capitalize words, unless they start with an x, in which case we lowercase the x and uppercase 2nd letter

#     # Rename the remaining columns
#     remaining_columns = [col for col in players_only_df.columns if col not in per90_columns]
#     rename_dict_rest = {
#         column: " ".join(
#             word.title() if not word.startswith('x') else 'x' + word[1:].capitalize() 
#             for word in column.replace('_', ' ').split()
#         ) for column in remaining_columns
#     }
#     players_only_df.rename(columns=rename_dict_rest, inplace=True)

#     print(f"Printing players_df columns:\n{players_df.columns.tolist()}")

#     # create a list of the columns that we want to convert to numeric
#     players_only_df = players_only_df.apply(pd.to_numeric, errors='ignore')

#     # loop through the dataframe and make sub dataframes that are grouped by season
#     seasons = players_only_df['Season'].unique().tolist()
#     teams = players_only_df['Team'].unique().tolist()
#     vs_teams = players_only_df['Opponent'].unique().tolist()
#     ages = players_only_df['Age'].unique().tolist()
#     nations = players_only_df['Nationality'].unique().tolist()
#     positions = players_only_df['Position'].unique().tolist()
#     referees = players_only_df['Referee'].unique().tolist()
#     venues = players_only_df['Venue'].unique().tolist()

#     # create list of objects above
#     categories = [seasons, teams, vs_teams, ages, nations, positions, referees, venues]

#     # create a dictionary of dataframes, one for each season
#     season_dfs = {}
#     teams_dfs = {}
#     vs_teams_dfs = {}
#     ages_dfs = {}
#     nations_dfs = {}
#     positions_dfs = {}
#     referees_dfs = {}
#     venues_dfs = {}

#     # for i in categories:
#     #     if i:
#     #         for j in i:
#     #             if j:
#     #                 for k in i:
#     #                     if k:
#     #                         season_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         teams_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         vs_teams_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         ages_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         nations_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         positions_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         referees_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
#     #                         venues_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]



#     for i in enumerate(categories):
#         if i[1]:
#             for j in i[1]:
#                 # create a dictionary as such but in a way thats iterable
#                 season_dfs[j] = players_only_df[players_only_df['Season'] == j]
#                 teams_dfs[j] = players_only_df[players_only_df['Team'] == j]
#                 vs_teams_dfs[j] = players_only_df[players_only_df['Opponent'] == j]
#                 ages_dfs[j] = players_only_df[players_only_df['Age'] == j]
#                 nations_dfs[j] = players_only_df[players_only_df['Nationality'] == j]
#                 positions_dfs[j] = players_only_df[players_only_df['Position'] == j]
#                 referees_dfs[j] = players_only_df[players_only_df['Referee'] == j]
#                 venues_dfs[j] = players_only_df[players_only_df['Venue'] == j]

#     stat = 'Pass xA Per90'
#     x_cat = 'Nationality'
        
#     for season in seasons:
#         season_data = players_only_df[players_only_df['Season'] == season]

#             # Calculate value counts for 'Nationality' and select top 20
#         top_nationalities = season_data[x_cat].value_counts().head(20).index

#         # Filter data for the top 20 nationalities
#         filtered_data = season_data[season_data[x_cat].isin(top_nationalities)]

#         # get the top 20 of each x that will be selected so that figure stays consistent
        
#         # teams_dfs[team] = players_only_df[players_only_df['team'] == team]

#         fig = px.scatter(
#             filtered_data,
#             x=x_cat,
#             y=stat,
#             color=stat,
#             color_continuous_scale='reds',
#             hover_data=['Player'],
            
#         )

#         st.info(f"Displaying {season} data for {stat} by {x_cat}")
#         st.plotly_chart(fig, theme="streamlit", use_container_width=True)

#         return players_only_df, season_dfs, teams_dfs, vs_teams_dfs, ages_dfs, nations_dfs, positions_dfs, referees_dfs, venues_dfs

def process_and_reorder_df(chrono_team_df, selected_team, selected_opponent):

    def compute_win_draw_loss(df):
        df['selected_team_draw'] = np.where(df['selected_team_score'] == df['selected_opponent_score'], 1, 0)
        df['selected_team_won'] = np.where(df['selected_team_score'] > df['selected_opponent_score'], 1, 0)
        df['selected_team_lost'] = np.where(df['selected_team_score'] < df['selected_opponent_score'], 1, 0)
        df['selected_opponent_draw'] = np.where(df['selected_opponent_score'] == df['selected_team_score'], 1, 0)
        df['selected_opponent_won'] = np.where(df['selected_opponent_score'] > df['selected_team_score'], 1, 0)
        df['selected_opponent_lost'] = np.where(df['selected_opponent_score'] < df['selected_team_score'], 1, 0)
        return df

    def compute_xg(df):
        df['selected_team_xg'] = np.where(df['selected_team'] == df['home_team'], df['home_xg'], df['away_xg'])
        df['selected_opponent_xg'] = np.where(df['selected_opponent'] == df['home_team'], df['home_xg'], df['away_xg'])
        return df

    def compute_result_categories(df):
        team_conditions = [(df['selected_team_won'] == 1), (df['selected_team_lost'] == 1), (df['selected_team_draw'] == 1)]
        opponent_conditions = [(df['selected_opponent_won'] == 1), (df['selected_opponent_lost'] == 1), (df['selected_opponent_draw'] == 1)]
        cat_choices = ['W', 'L', 'D']
        binary_choices = [1, -1, 0]

        df['selected_team_result_category'] = np.select(team_conditions, cat_choices, default=np.nan)
        df['selected_opponent_result_category'] = np.select(opponent_conditions, cat_choices, default=np.nan)
        df['selected_team_normalized_result'] = np.select(team_conditions, binary_choices, default=np.nan)
        df['selected_opponent_normalized_result'] = np.select(opponent_conditions, binary_choices, default=np.nan)
        return df

    def compute_home_teams(df):
        df['selected_team_home'] = np.where(df['selected_team'] == df['home_team'], 1, 0)
        df['selected_opponent_home'] = np.where(df['selected_opponent'] == df['home_team'], 1, 0)
        return df

    def reorder_df(df):
        new_columns = ['selected_team', 'selected_opponent', 'selected_team_score', 'selected_opponent_score', 'selected_team_draw', 'selected_team_won', 'selected_team_lost', 'selected_opponent_won', 'selected_opponent_lost', 'selected_team_xg', 'selected_opponent_xg', 'selected_team_result_category', 'selected_opponent_result_category', 'selected_team_normalized_result', 'selected_opponent_normalized_result', 'selected_team_home', 'selected_opponent_home']
        id_columns = ['matchup_id', 'season_matchup_id']
        rest_of_cols = [col for col in df.columns if col not in new_columns + id_columns]
        df = df[id_columns + new_columns + rest_of_cols]
        return df
    
    # Main operations
    chrono_team_df['selected_team'] = selected_team
    chrono_team_df['selected_opponent'] = selected_opponent
    chrono_team_df['selected_team_score'] = np.where(chrono_team_df['selected_team'] == chrono_team_df['home_team'], chrono_team_df['home_score'], chrono_team_df['away_score'])
    chrono_team_df['selected_opponent_score'] = np.where(chrono_team_df['selected_opponent'] == chrono_team_df['home_team'], chrono_team_df['home_score'], chrono_team_df['away_score'])
    
    chrono_team_df = compute_win_draw_loss(chrono_team_df)
    chrono_team_df = compute_xg(chrono_team_df)
    chrono_team_df = compute_result_categories(chrono_team_df)
    chrono_team_df = compute_home_teams(chrono_team_df)
    chrono_team_df = reorder_df(chrono_team_df)
    
    return chrono_team_df

def create_multiselect_seasons(players_df):
    """
    Summary:
        This function creates a multiselect for the seasons.

    Args:
        players_df (pandas DataFrame): The players df.

    Returns:
        selected_seasons (list): The selected seasons.
    """
    print(f"Running create_multiselect_seasons()...")
    # Get all unique seasons from the dataframe
    seasons = sorted(players_df["season"].unique())

    # Create a multiselect for the seasons with default selected as all seasons
    selected_seasons = st.multiselect(
        "Select Season(s)", seasons, default=seasons, key="seasons", 
        help="Please select at least one season.")

    if not selected_seasons:
        st.warning("Please select at least one season.")
        return
    
    # filter the dataframe to only include the selected seasons
    filtered_df = players_df[players_df["season"].isin(selected_seasons)]
    
    print(f"<> Printing unique seasons' values based on user-selected seasons:\n   ---------------------------------------------------------------\n   == {filtered_df['season'].unique()}")

    return selected_seasons, filtered_df

def create_dropdown_teams(filtered_df):
    """
    Summary:
        This function creates a dropdown for the user to select team and opponent.
        Based on the selected_seasons list, this function creates a dropdown for the teams from those seasons.
        This function also sets the default selected team to "Manchester United" and the default selected opponent to "Liverpool".
        Column match_teams and season_match_teams is created as such: 
        ``` players_df['match_teams'] = ['_'.join(sorted(map(str, row))) for row in zip(players_df['team'], players_df['opponent'])]
            players_df['season_match_teams'] = players_df['match_teams'] + '_' + players_df['season'].astype(str)

            # make sure there is no whitespace in match_teams and season_match_teams
            players_df['match_teams'] = players_df['match_teams'].str.replace(' ', '_')
            players_df['season_match_teams'] = players_df['season_match_teams'].str.replace(' ', '_')

            # convert to string
            players_df['match_teams'] = players_df['match_teams'].astype(str)
            players_df['season_match_teams'] = players_df['season_match_teams'].astype(str)
        ```
        We can create an identical string in this function from the selected_team and selected_opponent.
        By doing this 


    Args:
        players_df (pandas DataFrame): The players df.
        selected_seasons (list): The selected seasons.

    Returns:
        selected_teams (str): The selected team.
        selected_opponent (str): The selected opponent.
        filtered_df (pandas DataFrame): The filtered df.
    """
    print(f"<> Running create_dropdown_teams()...")

    # Filter the DataFrame based on selected seasons 
    if filtered_df is not None:  # Only if there are any selected seasons
        # print unique seasons' values based on unique seasons in filtered_df
        print(f"<> Printing unique seasons' values based on unique seasons in filtered_df:\n   ---------------------------------------------------------------\n   == {filtered_df['season'].unique()}")

    teams = sorted(filtered_df["team"].unique())

    # Make sure "Manchester United" is the first item in the list
    if "Manchester United" in teams:
        teams.remove("Manchester United")
        teams.insert(0, "Manchester United")

    # Create a dropdown for the teams
    selected_team = st.selectbox(
        "Select Team", teams, key="teams",
        help="Please select a team.")

    # Create a dropdown for the opponents
    opponents = [team for team in teams if team != selected_team]

    # Make sure "Liverpool" is the first item in the list
    if "Liverpool" in opponents:
        opponents.remove("Liverpool")
        opponents.insert(0, "Liverpool")

    selected_opponent = st.selectbox(
        "Select Opponent", opponents, key="opponents",
        help="Please select an opponent.")
    
    print(f"<> Printing selected_team: == {selected_team}")
    print(f"<> Printing selected_opponent: == {selected_opponent}")

    return selected_team, selected_opponent, filtered_df

def filter_df_by_team_and_opponent(df, selected_team, selected_opponent):

    print(f"<> Running filter_df_by_team_and_opponent()...")

    id_columns = ['matchup_id', 'season_matchup_id']

    # Filter the df by the selected_team and selected_opponent
    filtered_df = df[((df["team"] == selected_team) | (df["team"] == selected_opponent)) & 
        ((df["opponent"] == selected_team) | (df["opponent"] == selected_opponent))]
    
    filtered_df = filtered_df.sort_values(by=['date'], ascending=True)

    # print additional details this dataframe
    print(f"<> This dataframe is made up of the player and match statistics for {selected_team} vs {selected_opponent} from season {filtered_df['season_long'].unique()[0]} to {filtered_df['season_long'].unique()[-1]}.\n  ---------------------------------------------------------------\n<> Over that time {df['player'].nunique()} players have played in this fixture, with {filtered_df[filtered_df['team'] == selected_team]['player'].nunique()} players playing for {selected_team} and {filtered_df[filtered_df['team'] == selected_opponent]['player'].nunique()} players playing for {selected_opponent}.\n  ---------------------------------------------------------------\n{filtered_df['player'].value_counts().head(10)}")

    updated_df = process_and_reorder_df(filtered_df, selected_team, selected_opponent)

    # sort by date
    updated_df = updated_df.sort_values(by=['date'], ascending=True)

    grouped_player_df = updated_df.groupby(['player', 'season', 'team', 'matchup_id', 'season_matchup_id', 'selected_team_score', 'selected_opponent_score', 'selected_team_draw', 'selected_team_won', 'selected_team_lost', 'selected_opponent_won', 'selected_opponent_lost', 'selected_team_xg', 'selected_opponent_xg', 'selected_team_result_category', 'selected_opponent_result_category', 'selected_team_normalized_result', 'selected_opponent_normalized_result', 'selected_team_home', 'selected_opponent_home', 'home_team', 'away_team']).agg({'selected_team': 'first', 'selected_opponent': 'first'}).reset_index()

    selected_teams_df = updated_df.groupby(['matchup_id', 'season_matchup_id', 'selected_team_score', 'selected_opponent_score', 'selected_team_draw', 'selected_team_won', 'selected_team_lost', 'selected_opponent_won', 'selected_opponent_lost', 'selected_team_xg', 'selected_opponent_xg', 'selected_team_result_category', 'selected_opponent_result_category', 'selected_team_normalized_result', 'selected_opponent_normalized_result', 'selected_team_home', 'selected_opponent_home', 'date', 'home_team', 'away_team']).agg({'selected_team': 'first', 'selected_opponent': 'first'}).reset_index()

    # print additional details this dataframe
    # st.info("Checking grouped_player_df...")
    # grouped_player_df.shape

    # st.info("Checking selected_teams_df...")
    selected_teams_df = selected_teams_df.copy()

    selected_teams_df = selected_teams_df.sort_values(by=['date'], ascending=True)

    return selected_teams_df, grouped_player_df

def get_teams_stats(selected_teams_df, selected_team, selected_opponent):
    """
    Summary:
        This function returns the teams stats for the selected team and opponent using vectorized computations.
        These stats are:
        - Total Games
        - Total Wins
        - Total Losses
        - Total Draws
        - Total Goals Scored
        - Total Goals Conceded
        - xG For
        - xG Against
        - Clean Sheets

    Args:
        df (pandas DataFrame): The df to be filtered.
        team (str): The selected team.
        opponent (str): The selected opponent.

    Returns:
        teams_stats (dict): The teams stats.
    """

    teams = [selected_team, selected_opponent]
    
    team_stats_df = pd.DataFrame()

    for team in teams:

        team_row = selected_teams_df[(selected_teams_df['selected_team'] == team) | (selected_teams_df['selected_opponent'] == team)]

        total_games = team_row.shape[0]
        total_wins = team_row[team_row['selected_team_result_category'] == 'W'].shape[0]
        total_draws = team_row[team_row['selected_team_result_category'] == 'D'].shape[0]
        total_losses = team_row[team_row['selected_team_result_category'] == 'L'].shape[0]
        goals_scored = team_row['selected_team_score'].sum()
        goals_conceded = team_row['selected_opponent_score'].sum()
        xG_for = team_row['selected_team_xg'].sum()
        xG_against = team_row['selected_opponent_xg'].sum()
        clean_sheets = team_row[team_row['selected_opponent_score'] == 0].shape[0]

        team_stats = pd.Series({'Total Games': total_games, 'Total Wins': total_wins, 'Total Losses': total_losses, 
                                'Total Draws': total_draws, 'Total Goals Scored': goals_scored, 
                                'Total Goals Conceded': goals_conceded, 'xG For': xG_for, 'xG Against': xG_against, 
                                'Clean Sheets': clean_sheets}, name=team)
        team_stats_df = team_stats_df.append(team_stats)

    team_stats_og = team_stats_df.copy()

    team_stats_df = team_stats_df.T
    team_stats_df = team_stats_df.round(2)

    return team_stats_og, team_stats_df

def display_quant_stats(selected_teams_df, selected_team, selected_opponent):
    def format_dataframe(team_df, team, prefix, opponent_prefix):
        stats_dict = {}
        for stat in ['score', 'xg']:
            stats_dict[stat] = [team_df[prefix + stat].tolist(), team_df[prefix + stat].sum(), team_df[prefix + stat].mean()]
        
        stats_dict['clean_sheets'] = [(team_df[opponent_prefix + 'score'] == 0).astype(int).tolist(), 
                                       (team_df[opponent_prefix + 'score'] == 0).astype(int).sum(), 
                                       (team_df[opponent_prefix + 'score'] == 0).astype(int).mean()]
        
        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index').round(2)
        team_sum = f'Sum ({team})'
        team_mean = f'Mean ({team})'
        stats_df.columns = [team, team_sum, team_mean]

        return stats_df

    # Sort by date before creating the stats
    selected_teams_df = selected_teams_df.sort_values(by='date', ascending=True)

    team_stats_df = format_dataframe(selected_teams_df, selected_team, 'selected_team_', 'selected_opponent_')
    opponent_stats_df = format_dataframe(selected_teams_df, selected_opponent, 'selected_opponent_', 'selected_team_')

    combined_df = pd.concat([team_stats_df, opponent_stats_df], axis=1)

    linechart_df = combined_df.drop(columns=[f'Sum ({selected_team})', f'Mean ({selected_team})', f'Sum ({selected_opponent})', f'Mean ({selected_opponent})'])

    linechart_df = linechart_df.rename(index={'score': 'Goals', 'xg': 'xG', 'clean_sheets': 'Clean Sheets'})
    
    st.dataframe(
        linechart_df,
        column_config={
            selected_team: st.column_config.LineChartColumn(
                f"Stats Over Time ({selected_team})",
                width="medium",
                help=f"Line chart of {selected_team}'s stats over time",
            ),
            selected_opponent: st.column_config.LineChartColumn(
                f"Stats Over Time ({selected_opponent})",
                width="medium",
                help=f"Line chart of {selected_opponent}'s stats over time",
            ),
        },
        use_container_width=True,
    )

    sum_mean_df = combined_df.drop(columns=[selected_team, selected_opponent])
    sum_mean_df = sum_mean_df.reindex(columns=[f'Mean ({selected_team})', f'Mean ({selected_opponent})', f'Sum ({selected_team})', f'Sum ({selected_opponent})'])

    # rename the indexes to be more readable
    sum_mean_df = sum_mean_df.rename(index={'Goals Scored': 'Goals', 'xg': 'xG', 'clean_sheets': 'Clean Sheets'})

    # st.info(f"**{selected_team}** stats over time:")
    st.dataframe(sum_mean_df, use_container_width=True)

def display_qual_stats(selected_teams_df, selected_team, selected_opponent):

    def get_qual_stats_over_time(df, team, opponent):
        # Filter dataframe based on selected teams
        condition = (df['selected_team'] == team) & (df['selected_opponent'] == opponent)
        team_df = df[condition].copy()

        # Check if data exists
        if not team_df.empty:
            # Extract normalized results and clean sheets into a list
            team_results = team_df['selected_team_normalized_result'].tolist()
            opponent_results = team_df['selected_opponent_normalized_result'].tolist()

            # Compute clean sheets over time (1 if clean sheet, 0 otherwise)
            team_clean_sheets = (team_df['selected_opponent_score'] == 0).astype(int).tolist()
            opponent_clean_sheets = (team_df['selected_team_score'] == 0).astype(int).tolist()

            # Put these lists into a DataFrame
            results_df = pd.DataFrame({
                selected_team: [team_results, team_clean_sheets],
                selected_opponent: [opponent_results, opponent_clean_sheets]
            }, index=['Results Over Time', 'Clean Sheets'])
        else:
            results_df = pd.DataFrame()

        return results_df, team_df
    
    def format_special_qual_columns(df):
        if not df.empty:
            st.dataframe(
                df,
                column_config={
                    selected_team: st.column_config.BarChartColumn(
                        f"{selected_team} Matches Results Data Over Time",
                        help="Normalized  Matches Results Data Over Time for " + selected_team,
                        y_max=1.0,
                        y_min=-1.0                    ),
                    selected_opponent: st.column_config.BarChartColumn(
                        f"{selected_opponent} Matches Results Data Over Time",
                        help="Normalized  Matches Results Data Over Time for " + selected_opponent,
                        y_max=1.0,
                        y_min=-1.0
                    )
                },
                use_container_width=True,
            )
        else:
            st.warning(f"No data available for {selected_team} vs {selected_opponent}.")

    results_df, team_df = get_qual_stats_over_time(selected_teams_df, selected_team, selected_opponent)

    format_special_qual_columns(results_df)

def get_players_stats(player_level_df, selected_team, selected_opponent):
    stats_categories = ['nationality', 'age', 'minutes', 'goals', 'assists', 'pens_made', 'pens_att', 'shots', 'shots_on_target', 'cards_yellow', 'cards_red', 'touches', 'tackles', 'interceptions', 'blocks', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'passes_completed', 'passes', 'passes_pct', 'progressive_passes', 'carries', 'progressive_carries', 'take_ons', 'take_ons_won', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocked_shots', 'blocked_passes', 'tackles_interceptions', 'clearances', 'errors', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries_distance', 'carries_progressive_distance', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

    selected_stats_categories = st.multiselect("Select stats categories", stats_categories, default=['gca', 'npxg', 'xg_assist', 'pass_xa'])

    for team in [selected_team, selected_opponent]:
        st.title(team)  # Set the title to the current team

        team_condition = (player_level_df['team'] == team)

        for stat_category in selected_stats_categories:
            # Get the stats for the current team
            team_player_stats = player_level_df[team_condition].groupby(['player', 'team'])[stat_category].agg(['sum', 'mean']).reset_index()

            st.subheader(stat_category)  # Set the subtitle to the current stat category

            # Display the stats of the top 5 unique players
            team_player_stats = team_player_stats.sort_values(by=['sum', 'mean'], ascending=False).head(5)

            # drop the team column
            team_player_stats = team_player_stats.drop(columns=['team'])

             # round relevant columns
            team_player_stats = team_player_stats.round({'sum': 2, 'mean': 2})

            # Apply color gradients
            # Display styled dataframe
            display_styled_dataframe_v2(team_player_stats)

            # display_styled_dataframe(team_player_stats, 'sum', 'mean')

def show_teams_stats(team_stats_df, cm):
    
    display_styled_dataframe_simple(team_stats_df, cm)

def calculate_per90(df):
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Remove 'season' from the list of numeric columns if it's there
    numeric_cols = [col for col in numeric_cols if col not in ['season']]

    # Define aggregation dictionary for summing numeric columns and getting first observation for categorical
    agg_dict = {col: 'sum' for col in numeric_cols}
    agg_dict.update({col: 'first' for col in categorical_cols})

    # Group by player and season, and aggregate
    df_grouped = df.groupby(['player', 'season']).agg(agg_dict)

    # Define the columns you want to calculate per90s for
    per90_cols = numeric_cols  # adjust this list according to your needs

    # Calculate per90s
    for col in per90_cols:
        if col in df_grouped.columns:
            df_grouped[col+'_per90'] = df_grouped[col] / (df_grouped['minutes'] / 90)
    
    # Create a new dataframe with only per90 columns and 'player', 'season'
    df_per90 = df_grouped.filter(regex='per90|Player|Season')

    return df_per90

def rename_columns(df):
    """
    Rename the columns of a DataFrame: replace underscores with spaces, capitalize words, and keep the letter 'x' lowercase.
    """
    rename_dict = {
        column: " ".join(
            word.title() if not word.startswith('x') else 'x' + word[1:].capitalize() 
            for word in column.split('_')
        ) for column in df.columns
    }
    
    return df.rename(columns=rename_dict)

def create_stat_dropdown(stat_name, stat_list, current_stat, default_stat):
    """
    Creates a dropdown menu for the given stat and returns the selected stat.
    """
    if current_stat not in stat_list:
        current_stat = default_stat
    return st.sidebar.selectbox(f'Select {stat_name}', stat_list, index=stat_list.index(current_stat))

def dropdown_for_stat_selection(label, stats, selected_stat):
    """Helper function for stats dropdown creation."""
    if selected_stat not in stats:
        selected_stat = stats[0]
    return st.sidebar.selectbox(label, stats, index=stats.index(selected_stat))

def dropdown_for_player_stats(players_only_df, selected_player, selected_season, *selected_stats):
    # Code related to players and seasons stays the same...

    exclude_stats_list = [
        'Matches Played', 'Rk', 'Born', 'Age', 'Games Played', 'Minutes Played', 'Player', 
        'Nation', 'Pos', 'Team', 'Matches', 'Position Category', 'League', 'Season', '90S'
    ]

    stats = players_only_df.select_dtypes(exclude=['object']).columns.tolist()
    stats = [stat for stat in stats if stat not in exclude_stats_list]

    selected_stats = [
        dropdown_for_stat_selection(f'Select Stat {i+1}', stats, selected_stat)
        for i, selected_stat in enumerate(selected_stats)
    ]

    return selected_player, selected_season, *selected_stats

def dropdown_for_player_stats(players_only_df, selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5):

    # create a list of all the players, sorted by team, then season
    players = players_only_df.sort_values(by=['Team', 'Season'])['Player'].unique().tolist()

    # create a dropdown menu for the players
    selected_player = st.sidebar.selectbox('Select Player', players, index=players.index(selected_player))

    # we should only display seasons that the player has played in

    seasons = players_only_df[players_only_df['Player'] == selected_player]['Season'].unique().tolist()

    print(f"Printing seasons: {seasons}")

    print(f"Printing selected season: {selected_season}")

    if selected_season not in seasons:
        selected_season = seasons[0]

    # create a dropdown menu for the seasons
    selected_season = st.sidebar.selectbox('Select Season', seasons, index=seasons.index(selected_season))

    # selected_season = st.slider('Select Season', min_value=min(seasons), max_value=max(seasons), value=selected_season, step=1)

    all_seasons = st.sidebar.checkbox('All Seasons', value=False)
    all_seasons_selected = all_seasons  # New variable to determine if all seasons are selected

    if all_seasons:
        selected_season = seasons[0]

    exclude_stats_list = ['Matches Played', 'Rk', 'Born', 'Age', 'Games Played', 'Minutes Played', 'Player', 'Nation', 'Pos', 'Team', 'Matches', 'Position Category', 'League', 'Season', '90S']

    # get stats that are not in that list and that are not of type object

    stats = players_only_df.select_dtypes(exclude=['object']).columns.tolist()

    stats = [stat for stat in stats if stat not in exclude_stats_list]

    # # create a list of all the stats from the numeric columns
    # stats = players_only_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    print(f"Printing stats: {stats}")

    # create a dropdown menu for the stats iterating through and checking if stat is in the list of stats
    if selected_stat1 not in stats:
        print(f"selected_stat1: {selected_stat1} not in stats: {stats}")
        selected_stat1 = stats[0]
        selected_stat1 = st.sidebar.selectbox('Select Stat 1', stats, index=stats.index(selected_stat1))
    else:
        selected_stat1 = st.sidebar.selectbox('Select Stat 1', stats, index=stats.index(selected_stat1))

    if selected_stat2 not in stats:
        print(f"selected_stat2: {selected_stat2} not in stats: {stats}")
        selected_stat2 = stats[0]
        selected_stat2 = st.sidebar.selectbox('Select Stat 2', stats, index=stats.index(selected_stat2))
    else:
        selected_stat2 = st.sidebar.selectbox('Select Stat 2', stats, index=stats.index(selected_stat2))

    if selected_stat3 not in stats:
        print(f"selected_stat3: {selected_stat3} not in stats: {stats}")
        selected_stat3 = stats[0]
        selected_stat3 = st.sidebar.selectbox('Select Stat 3', stats, index=stats.index(selected_stat3))
    else:
        selected_stat3 = st.sidebar.selectbox('Select Stat 3', stats, index=stats.index(selected_stat3))

    if selected_stat4 not in stats:
        print(f"selected_stat4: {selected_stat4} not in stats: {stats}")
        selected_stat4 = stats[0]
        selected_stat4 = st.sidebar.selectbox('Select Stat 4', stats, index=stats.index(selected_stat4))
    else:
        selected_stat4 = st.sidebar.selectbox('Select Stat 4', stats, index=stats.index(selected_stat4))

    if selected_stat5 not in stats:
        print(f"selected_stat5: {selected_stat5} not in stats: {stats}")
        selected_stat5 = stats[0]
        selected_stat5 = st.sidebar.selectbox('Select Stat 5', stats, index=stats.index(selected_stat5))
    else:
        selected_stat5 = st.sidebar.selectbox('Select Stat 5', stats, index=stats.index(selected_stat5))


    return selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5, seasons, all_seasons_selected

def normalize_encoding(df, column_name):
    """
    Normalizes the encoding of a pandas DataFrame column using the unidecode library.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the column to normalize.
        column_name (str): The name of the column to normalize.

    Returns:
        pandas DataFrame: A copy of the original DataFrame with the specified column normalized.
    """
    # Create a copy of the original DataFrame
    normalized_df = df.copy()

    # Normalize the encoding of the specified column using unidecode
    normalized_df[column_name] = normalized_df[column_name].apply(unidecode)

    return normalized_df

def process_player_data(players_only_df):
    # 1. Set default values
    default_stats = [
        'Shot Creating Actions', 
        'Assists Plus Expected Assisted Goals', 
        'Progressive Passes Received', 
        'Passes Into Penalty Area', 
        'Dead Balls Leading To Shots'
    ]

    # 2. Get values from the dropdowns
    selected_data = dropdown_for_player_stats(players_only_df, 'Bruno Fernandes', '2023', *default_stats)
    selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5, seasons, all_seasons_selected = selected_data

    # 3. Ensure uniqueness of selected stats
    def ensure_unique_stats(stats_list, default_list):
        for i, stat in enumerate(stats_list):
            while stats_list.count(stat) > 1:
                next_index = (default_list.index(stat) + 1) % len(default_list)
                stat = default_list[next_index]
            stats_list[i] = stat
        return stats_list

    selected_stats = [selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5]
    selected_stats = ensure_unique_stats(selected_stats, default_stats)
    selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = selected_stats

    # 4. Convert the season column and selected season to integers
    players_only_df['Season'] = players_only_df['Season'].astype(int)
    selected_season = int(selected_season)

    # 5. Handle column normalization
    if 'player' in players_only_df.columns.tolist():
        players_only_df = players_only_df.rename(columns={'player': 'Player'})
    if 'season' in players_only_df.columns.tolist():
        players_only_df = players_only_df.rename(columns={'season': 'Season'})

    # 6. Filter data based on selections
    if all_seasons_selected:
        player_df = players_only_df[players_only_df['Player'] == selected_player]
    elif selected_season in seasons:
        player_df = players_only_df[(players_only_df['Player'] == selected_player) & (players_only_df['Season'] == selected_season)]
    else:
        selected_season = seasons[0]
        player_df = players_only_df[(players_only_df['Player'] == selected_player) & (players_only_df['Season'] == selected_season)]

    # 7. Prepare data for the chart
    stats_values = [
        player_df[selected_stat1].values[0],
        player_df[selected_stat2].values[0],
        player_df[selected_stat3].values[0],
        player_df[selected_stat4].values[0],
        player_df[selected_stat5].values[0]
    ]

    # Adding a radio button for per 90 stats
    per_90 = st.sidebar.radio('Choose display type:', ('Per 90 Stats', 'Raw Data'))

    # Inside the function, after you've selected the required data:
    if per_90 == 'Per 90 Stats':
        stats_values = [
            player_df[f"{selected_stat1} Per90"].values[0],
            player_df[f"{selected_stat2} Per90"].values[0],
            player_df[f"{selected_stat3} Per90"].values[0],
            player_df[f"{selected_stat4} Per90"].values[0],
            player_df[f"{selected_stat5} Per90"].values[0]
        ]

    # Ensure stats_values is a one-dimensional array
    stats_values = np.array(stats_values).flatten()

    ranks = [sorted(stats_values, reverse=True).index(val) + 1 for val in stats_values]  # This gives ranks in descending order

    # 8. Plot chart using Plotly
    st.info(f"Displaying {selected_player}'s stats for {selected_season}")

    fig2 = px.bar_polar(
    player_df,
    r=stats_values,
    theta=selected_stats,
    color=selected_stats,
    color_discrete_sequence=px.colors.sequential.Plasma_r,
    title=f"{selected_player}'s Stats"
    # template='plotly_dark'
    )

    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(stats_values) + 0.05]
            )
        ),
        bargap=0.6,
        bargroupgap=0.1,
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        showlegend=False,  # Hide the legend
        hoverlabel=dict(
            font=dict(
                color='black',  # Setting hover label font color to dark
                size=15  # Increase font size for better visibility
            ),
            bgcolor='rgba(255,255,255,0.7)'  # Setting hover label background color to semi-transparent white
        )
    )

    # Data trace adjustments
    fig2.data[0].update(
        marker_line_color="black",
        marker_line_width=2,
        opacity=0.8,
        width=1,
        hovertemplate="<b>%{theta}</b><br><br>" + "%{r}<br>" + "<extra></extra>",
        customdata=ranks  # Pass the ranks as custom data

    )

    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

    # # create a five point chart for the players top 5 per90 stats
    # fig = px.line_polar(
    #     player_df,
    #     r=stats_values,
    #     theta=[selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5],
    #     line_close=True,
    #     title=f"{selected_player}'s Stats",
    #     color_discrete_sequence=px.colors.sequential.Plasma_r,
    #     color=[selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5],
    #     template='plotly_dark',
    #     render_mode='svg'

    # )
    # st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # create a dataframe for the selected player
    st.write(stats_values)

    return player_df

def min_max_scale(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Initialize a scaler
    scaler = MinMaxScaler()

    # Replace infinities with NaN, then NaN with 0
    df[numeric_cols].replace([np.inf, -np.inf], np.nan, inplace=True)
    df[numeric_cols].fillna(0, inplace=True)

    # Fit and transform the data
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def convert_to_int(item):
    if isinstance(item, list):
        return [int(i) for i in item]
    else:
        return int(item)
    
def parse_team_stats_table(table):
    try:
        column_headers = []
        data = []

        rows = table.find_all('tr')
        if rows:
            first_header_row_processed = False
            for i, row in enumerate(rows):
                # Ignore over_header rows and the last row which is the total
                if 'over_header' in row.get('class', []) or i == len(rows) - 1:
                    continue

                headers = row.find_all('th')
                if headers and not first_header_row_processed:
                    # Check if the th tags are direct children of the table
                    if headers[0].parent.name == 'tr' and headers[0].parent.parent.name == 'thead':
                        # Extract column headers
                        column_headers = [header.get(
                            'data-stat') for header in headers]
                        # Check if column_headers are unique
                        if len(column_headers) != len(set(column_headers)):
                            raise ValueError(
                                "Duplicate column headers detected.")

                        first_header_row_processed = True
                    continue  # Skip the rest of the loop for this iteration

                # Process data rows
                # Changed to find both 'th' and 'td' cells
                cells = row.find_all(['th', 'td'])
                if cells:
                    output_row = {}

                    for i, header in enumerate(column_headers):
                        if i < len(cells):
                            output_row[header] = cells[i].text
                        else:
                            # Set to None if cell does not exist
                            output_row[header] = None

                    data.append(output_row)

        # Apply unidecode to player names
        for row in data:
            if 'player' in row:
                row['player'] = unidecode.unidecode(row['player'])

        print(f'Column Headers: {column_headers}')

        return column_headers, data
    except Exception as e:
        raise
    
# def scraping_current_fbref(categories_list):
#     all_season_data = []  # List to store dataframes for each season

#     season = 2023
#     print(f"Scraping season: {season}")
#     player_table = None
#     scraped_columns_base = []
    
#     for cat in enumerate(categories_list):
#         # Handle most recent season differently

#         url = f'https://fbref.com/en/comps/Big5/{cat}/players/Big-5-European-Leagues-Stats'
            
#         print(f"A. Scraping {cat} player data for {season} - {url}")
#         resp = requests.get(url).text
#         htmlStr = resp.replace('<!--', '')
#         htmlStr = htmlStr.replace('-->', '')
        
#         if cat == 'playingtime':
#             temp_df = pd.read_html(htmlStr, header=1)[0]
#         else:
#             temp_df = pd.read_html(htmlStr, header=1)[1]
            
#         temp_df = temp_df[temp_df['Rk'] != 'Rk']  # Remove duplicate headers
#         temp_df['Season'] = season  # Add season column
#         temp_df['Season'] = temp_df['Season'].str[-4:]

#         if player_table is None:
#             player_table = temp_df
#             scraped_columns_base = player_table.columns.tolist()
#         else:
#             new_columns = [col for col in temp_df.columns if col not in scraped_columns_base and col not in ['Player', 'Squad', 'Season']]
#             temp_df = temp_df[['Player', 'Squad', 'Season'] + new_columns]
#             player_table = pd.merge(player_table, temp_df, on=['Player', 'Squad', 'Season'], how='left')
#             scraped_columns_base += new_columns
        
#         print(f"Finished scraping {cat} data for {season}, DataFrame shape: {temp_df.shape}")        
#         print(f"After operations and/or merging, player_table shape: {player_table.shape}")


#     all_season_data.append(player_table)

#     # Concatenate all seasons data into one DataFrame
#     final_player_table = pd.concat(all_season_data, ignore_index=True)

#     return final_player_table

def scraping_current_fbref(categories_list, db_name='soccer_stats.db'):
    conn = sqlite3.connect(db_name)  # Open connection to SQLite db
    all_season_data = []

    season = 2023
    print(f"Scraping season: {season}")
    player_table = None
    scraped_columns_base = []

    for cat in categories_list:
        url = f'https://fbref.com/en/comps/Big5/{cat}/players/Big-5-European-Leagues-Stats'
        print(f"A. Scraping {cat} player data for {season} - {url}")
        resp = requests.get(url).text
        soup = BeautifulSoup(resp, 'html.parser')
        table = soup.find('table')
        
        # Extract data from table
        column_headers, data = parse_team_stats_table(table)
        
        # Create table in SQLite DB
        create_table_from_columns(conn, f'{cat}_stats', column_headers)
        insert_data_into_table(conn, f'{cat}_stats', data)

        temp_df = pd.DataFrame(data)

        temp_df['season'] = season
        if player_table is None:
            player_table = temp_df
            scraped_columns_base = player_table.columns.tolist()
            scraped_columns_base
            print(f"Scraped columns base: {scraped_columns_base}")
        else:
            new_columns = [col for col in temp_df.columns if col not in scraped_columns_base and col not in ['player', 'team', 'season']]
            scraped_columns_base
            print(f"Scraped columns base: {scraped_columns_base}")
            temp_df = temp_df[['player', 'team', 'season'] + new_columns]
            player_table = pd.merge(player_table, temp_df, on=['player', 'season'], how='left')
            scraped_columns_base += new_columns
        
        print(f"Finished scraping {cat} data for {season}, DataFrame shape: {temp_df.shape}")        
        print(f"After operations and/or merging, player_table shape: {player_table.shape}")

    all_season_data.append(player_table)
    final_player_table = pd.concat(all_season_data, ignore_index=True)

    conn.close()  # Close the connection
    return final_player_table

def create_table_from_columns(conn, table_name, columns):
    columns_str = ', '.join([f'"{col}" TEXT' for col in columns])
    c = conn.cursor()
    c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})")
    conn.commit()

def insert_data_into_table(conn, table_name, data):
    c = conn.cursor()
    for row in data:
        placeholders = ', '.join(['?'] * len(row))
        columns = ', '.join(row.keys())
        values = tuple(row.values())
        c.execute(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values)
    conn.commit()

def clean_age_column(df):
    # strip the last 4 characters from the age column and convert to numeric
    df['age'] = pd.to_numeric(df['age'].str[:-4])
    return df

def create_sidebar_multiselect(df, column_name, title='Select Options', default_all=True, key_suffix=None):
    # Get unique values from the specified column
    options = sorted(df[column_name].astype(str).unique())

    # Set the default selected options based on the 'default_all' flag
    default_options = options if default_all else [options[0]]

    # Create a unique key by combining the title and key_suffix
    unique_key = f"{title}_{key_suffix}" if key_suffix else None

    # Create a multiselect in the sidebar with the unique values
    selected_options = st.sidebar.multiselect(title, options, default=default_options, key=unique_key)

    # If no options are selected, default to all available options
    if not selected_options:
        st.sidebar.warning(f"You have not selected any {title}. Defaulting to 'All'.")
        selected_options = options

    return selected_options


def get_color(value, cmap):
    color_fraction = value
    rgba_color = cmap(color_fraction)
    brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
    
    # Adjust the brightness threshold
    text_color = 'white' if brightness < 0.75 else 'black'
    
    return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'


def style_dataframe(df, selected_columns):
    cm_sns_copper = cm.get_cmap('copper')
    object_cmap = cm.get_cmap('copper')  # Choose a colormap for object columns

    # Create an empty DataFrame with the same shape as df
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    for col in df.columns:
        if col == 'player':  # Skip the styling for the 'player' column
            continue
        if df[col].dtype in [np.float64, np.int64] and col in selected_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / range_val, cm_sns_copper))
        elif df[col].dtype == 'object':
            unique_values = df[col].unique().tolist()
            styled_df[col] = df[col].apply(lambda x: get_color(unique_values.index(x) / len(unique_values), object_cmap))
    return styled_df

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
def load_data():
    return process_data(df, df2)

def get_color_from_palette(value, palette_name='copper'):
    cmap = mpl_cm.get_cmap(palette_name)
    rgba_color = cmap(value)
    color_as_hex = mcolors.to_hex(rgba_color)
    return color_as_hex

def style_dataframe_v2(df, selected_columns):
    object_cmap = mpl_cm.get_cmap('copper')

    # Create an empty DataFrame with the same shape as df
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)

    if 'Pos' in df.columns:
        unique_positions = df['Pos'].unique().tolist()

        # Define the colors for the positions
        position_colors = {
            "D": "background-color: #3d0b4d;",  # Specific purple color for "D"
            "M": "background-color: #08040f",  # Assigned color for "M"
            "F": "background-color: #050255"   # Assigned color for "F"
        }

        # Apply the colors to the 'Pos' and 'Player' columns
        styled_df['Pos'] = df['Pos'].apply(lambda x: position_colors[x])
        styled_df['Player'] = df['Pos'].apply(lambda x: position_colors[x])

    for col in df.columns:
        if col in ['Player', 'Pos']:
            continue

        col_dtype = df[col].dtype
        unique_values = df[col].unique().tolist()

        if len(unique_values) <= 3:
            constant_colors = [get_color(i / 2, mpl_cm.get_cmap('copper')) for i in range(len(unique_values))]
            color_mapping = {val: color for val, color in zip(unique_values, constant_colors)}
            styled_df[col] = df[col].apply(lambda x: color_mapping[x])
        elif col_dtype in [np.float64, np.int64] and col in selected_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / range_val, mpl_cm.get_cmap('copper')))
        elif col_dtype == 'object':
            styled_df[col] = df[col].apply(lambda x: get_color(unique_values.index(x) / len(unique_values), object_cmap))

    return styled_df

def get_color_0(value, cmap):
    color_fraction = value
    rgba_color = cmap(color_fraction)
    brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
    text_color = 'white' if brightness < 0.7 else 'black'
    return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

# def get_color(value, cmap):
#     color_fraction = value
#     rgba_color = cmap(color_fraction)
#     brightness = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
    
#     # Adjust the brightness threshold
#     text_color = 'white' if brightness < 0.75 else 'black'
    
#     return f'color: {text_color}; background-color: rgba({",".join(map(str, (np.array(rgba_color[:3]) * 255).astype(int)))}, 0.7)'

def create_custom_cmap_1(*colors):
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return custom_cmap

# def style_dataframe_custom(df, selected_columns, custom_cmap=None):
#     if custom_cmap:
#         object_cmap = custom_cmap
#     else:
#         object_cmap = create_custom_cmap()  # Assuming create_custom_cmap is defined elsewhere

#     Team_cmap = plt.cm.get_cmap('copper')
#     styled_df = pd.DataFrame('', index=df.index, columns=df.columns)

#     position_column = 'Pos' if 'Pos' in df.columns else 'Position' if 'Position' in df.columns else None

#     if position_column:
#         position_colors = {
#             "D": "background-color: #6d597a",
#             "M": "background-color: #370617",
#             "F": "background-color: #03071e"
#         }
#         styled_df[position_column] = df[position_column].apply(lambda x: position_colors.get(x, ''))
#         if 'Player' in df.columns:
#             styled_df['Player'] = df[position_column].apply(lambda x: position_colors.get(x, ''))

#     for col in selected_columns:
#         if col in ['Player', position_column]:
#             continue

#         try:
#             unique_values = df[col].unique()
#         except AttributeError as e:
#             print(f"AttributeError occurred for column: {col}. Error message: {e}")
#             continue

#         if len(unique_values) <= 3:
#             constant_colors = ["color: #eae2b7", "color: #FDFEFE", "color: #FDFAF9"]
#             most_common_value, _ = Counter(df[col]).most_common(1)[0]
#             other_colors = [color for val, color in zip(unique_values, constant_colors[1:]) if val != most_common_value]
#             color_mapping = {most_common_value: constant_colors[0], **{val: color for val, color in zip([uv for uv in unique_values if uv != most_common_value], other_colors)}}
#             styled_df[col] = df[col].apply(lambda x: color_mapping.get(x, ''))
        
#         elif 'Team' in df.columns:
#             n = len(unique_values)
#             for i, val in enumerate(unique_values):
#                 norm_i = i / (n - 1) if n > 1 else 0.5
#                 styled_df.loc[df[col] == val, col] = get_color(norm_i, Team_cmap)
        
#         else:
#             min_val = float(df[col].min())
#             max_val = float(df[col].max())
#             styled_df[col] = df[col].apply(lambda x: f'color: {matplotlib.colors.to_hex(object_cmap((float(x) - min_val) / (max_val - min_val)))}' if min_val != max_val else '')

#     return styled_df


def style_dataframe_custom(df, selected_columns, custom_cmap="copper", inverse_cmap=False, is_percentile=False):
    """
    Style the DataFrame based on the selected columns and color map.
    
    :param df: DataFrame to style
    :param selected_columns: List of columns to style
    :param custom_cmap: Color map name
    :param inverse_cmap: Whether to inverse the color map
    :param is_percentile: Whether to use divergent color map for percentiles
    :return: DataFrame Styler object
    """
    object_cmap = plt.cm.get_cmap(custom_cmap)
    styled_df = pd.DataFrame()

    position_column = 'Position' if 'Position' in df.columns else None
    if position_column:
        position_colors = {
            "D": "background-color: #6d597a; color: white",
            "M": "background-color: #08071d; color: white",
            "F": "background-color: #370618; color: white"
        }
        styled_df[position_column] = df[position_column].apply(lambda x: position_colors.get(x, ''))

        if 'Player' in df.columns:
            styled_df['Player'] = df[position_column].apply(lambda x: position_colors.get(x, ''))

    for col in selected_columns:
        if col in ['Player', position_column]:
            continue

        col_data = df[col]

        try:
            col_data = col_data.astype(float)
            min_val = col_data.min()
            max_val = col_data.max()
        except ValueError:
            min_val = max_val = None

        unique_values = col_data.unique()

        if len(unique_values) <= 3:
            constant_colors = ["#140b04", "#1c1625", "#460202"]
            text_colors = ['white', 'white', 'white']

            most_common_list = Counter(col_data).most_common(1)
            if most_common_list:
                most_common_value, _ = most_common_list[0]
            else:
                most_common_value = None

            other_values = [uv for uv in unique_values if uv != most_common_value]

            color_mapping = {
                val: f"background-color: {color}; color: {text}" 
                for val, color, text in zip([most_common_value] + other_values, constant_colors, text_colors)
            }

            styled_df[col] = col_data.apply(lambda x: color_mapping.get(x, ''))
        elif min_val is not None and max_val is not None:
            if min_val != max_val:
                styled_df[col] = col_data.apply(
                    lambda x: get_color((1 - (x - min_val) / (max_val - min_val)) if inverse_cmap else (x - min_val) / (max_val - min_val), object_cmap)
                )

    return styled_df

def create_custom_cmap(*colors, base_cmap=None, brightness_limit=None):
    if colors:
        return LinearSegmentedColormap.from_list('custom_cmap', colors)
    elif base_cmap and brightness_limit:
        base = plt.cm.get_cmap(base_cmap)
        color_list = [base(i) for i in range(256)]
        color_list = [(r * brightness_limit, g * brightness_limit, b * brightness_limit, a) for r, g, b, a in color_list]
        return LinearSegmentedColormap.from_list(base_cmap, color_list)

def create_custom_sequential_cmap(*colors):
    return LinearSegmentedColormap.from_list('custom_sequential_cmap', colors)

def create_custom_cmap_0(base_cmap='magma', brightness_limit=1):
    base = plt.cm.get_cmap(base_cmap)
    color_list = [base(i) for i in range(256)]
    # Apply brightness limit
    color_list = [(r * brightness_limit, g * brightness_limit, b * brightness_limit, a) for r, g, b, a in color_list]
    return LinearSegmentedColormap.from_list(base_cmap, color_list)

def debug_dataframe(df, debug_message="Debugging DataFrame"):
    """
    Function to debug and evaluate DataFrame attributes.
    """
    print(f"\n{debug_message}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data Types:\n{df.dtypes}")

def rank_players_by_multiple_stats(df, stat_columns):
    """
    Ranks players based on multiple selected statistics.
    :param df: DataFrame containing player statistics
    :param stat_columns: List of column names of the statistics to rank by
    :return: DataFrame with added 'Rank' columns for each selected statistic
    """
    df = df.copy()
    for col in stat_columns:
        if col in df.columns:
            rank_col_name = f"{col}_Rank"
            ranks = df[col].rank(ascending=False, method='min')
            if ranks.isna().any():
                df[rank_col_name] = ranks
            else:
                df[rank_col_name] = ranks.astype(int)
    return df

@st.cache_data
def percentile_players_by_multiple_stats(df, stat_columns):
    """
    Shows players' statistics as percentiles based on multiple selected statistics.
    :param df: DataFrame containing player statistics
    :param stat_columns: List of column names of the statistics to show as percentiles
    :return: DataFrame with added 'Percentile' columns for each selected statistic
    """
    df = df.copy()
    for col in stat_columns:
        if col in df.columns:
            percentile_col_name = f"{col}_Pct"
            df[percentile_col_name] = df[col].apply(lambda x: percentileofscore(df[col], x))
    return df

def style_tp_dataframe_custom(df, selected_columns, custom_cmap_name="copper"):
    cmap = plt.cm.get_cmap(custom_cmap_name)
    
    styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
    
    position_column = 'Pos' if 'Pos' in df.columns else 'Position' if 'Position' in df.columns else None
    position_colors = {"D": "background-color: #6d597a", "M": "background-color: #370617", "F": "background-color: #03071e"}
    
    if position_column:
        styled_df[position_column] = df[position_column].map(position_colors.get)
        styled_df['Player'] = df[position_column].map(position_colors.get)

    for col in df.columns:
        if col in ['Player', position_column]:
            continue
        
        elif col == 'Team':
            team_rank = df['Team'].rank(method='min', ascending=False)
            max_rank = team_rank.max()
            styled_df['Team'] = team_rank.apply(lambda x: get_color((x - 1) / (max_rank - 1), cmap))

        elif len(df[col].unique()) <= 3:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            constant_colors = {"#eae2b7": "color: #eae2b7", "#FDFEFE": "color: #FDFEFE", "#FDFAF9": "color: #FDFAF9"}
            most_common_value, _ = Counter(df[col]).most_common(1)[0]
            color_mapping = {most_common_value: constant_colors["#eae2b7"]}
            styled_df[col] = df[col].map(color_mapping.get)

        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            min_val, max_val = df[col].min(), df[col].max()
            styled_df[col] = df[col].apply(lambda x: get_color((x - min_val) / (max_val - min_val), cmap) if max_val != min_val else '')

    return styled_df

def round_and_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value

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

def display_date_of_update(date_of_update):
    st.write(f'Last data refresh: {date_of_update}')

def round_and_format(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value

def load_csv(filepath):
    return pd.read_csv(filepath)

