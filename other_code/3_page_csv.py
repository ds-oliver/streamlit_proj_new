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

sys.path.append(os.path.abspath(os.path.join('./scripts')))

from constants import color1, color2, color3, color4, color5, cm

# function to load this csv /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/db_files/players.db and /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/csv_files/players.csv

file_path = 'data/data_out/final_data/csv_files/players.csv'
# if os.path.exists(file_path):
#     print(f"File {file_path} found.")
# else:
#     print(f"File {file_path} not found.")

@st.cache_resource
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

@st.cache_resource
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

@st.cache_resource
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

def process_player_data(players_df):
    """Separate the columns for player level analysis, we will then create new dataframes and dictionaries for different aggregations of the player data (percent of team total, percent of game total, percent of season, etc.), primarily using per90s"""

    players_only_df = players_df.copy()

    players_only_df.sort_values(by=['season', 'date', 'player'], ascending=True)

        # turn date to numerical value
    players_only_df['datetime'] = pd.to_datetime(players_only_df['date'])

    # groupby player and season but we need to keep the player column so we can merge back later

    # create per90 columns
    per90_columns = ['minutes', 'goals', 'assists', 'pens_made', 'pens_att', 'shots', 'shots_on_target', 'cards_yellow', 'cards_red', 'touches', 'tackles', 'interceptions', 'blocks', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'passes_completed', 'passes', 'passes_pct', 'progressive_passes', 'carries', 'progressive_carries', 'take_ons', 'take_ons_won', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocked_shots', 'blocked_passes', 'tackles_interceptions', 'clearances', 'errors', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries_distance', 'carries_progressive_distance', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

    per90_columns = ['minutes', 'goals', 'assists', 'pens_made', 'pens_att', 'shots', 'shots_on_target', 'cards_yellow', 'cards_red', 'touches', 'tackles', 'interceptions', 'blocks', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'passes_completed', 'passes', 'passes_pct', 'progressive_passes', 'carries', 'progressive_carries', 'take_ons', 'take_ons_won', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocked_shots', 'blocked_passes', 'tackles_interceptions', 'clearances', 'errors', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries_distance', 'carries_progressive_distance', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

    # create per90 minutes columns for all per90 columns
    for column in per90_columns:
        players_only_df[f"{column}_per90"] = players_only_df[column] / (players_only_df['minutes'] / 90)
        # round to 2 decimal places
        players_only_df[f"{column}_per90"] = players_only_df[f"{column}_per90"].round(2)

    # rename the columns so that they are more descriptive, underscores replaced with spaces and words capitalized, the letter x is always lowercase

    # first we do the per90 columns
    players_only_df = rename_columns(players_only_df)

    # 

    # rename_dict = {
    #     f"{column}_per90": " ".join(
    #         word.title() if not word.startswith('x') else 'x' + word[1:].capitalize() 
    #         for word in column.replace('_', ' ').split()
    #     ) + " Per90" for column in per90_columns
    # }
    # players_only_df.rename(columns=rename_dict, inplace=True)

    # # now we do the same formatting for other columns where we replace underscores with spaces and capitalize words, unless they start with an x, in which case we lowercase the x and uppercase 2nd letter

    # # Rename the remaining columns
    # remaining_columns = [col for col in players_only_df.columns if col not in per90_columns]
    # rename_dict_rest = {
    #     column: " ".join(
    #         word.title() if not word.startswith('x') else 'x' + word[1:].capitalize() 
    #         for word in column.replace('_', ' ').split()
    #     ) for column in remaining_columns
    # }
    # players_only_df.rename(columns=rename_dict_rest, inplace=True)

    print(f"Printing players_df columns:\n{players_only_df.columns.tolist()}")

    # create a list of the columns that we want to convert to numeric
    players_only_df = players_only_df.apply(pd.to_numeric, errors='ignore')

    # groupby player and season keeping all the columns


    # # groupby player and season keeping all the columns
    # players_only_df = players_only_df.groupby(['Player', 'Season', 'Team', 'Nationality', 'Opponent']).sum().reset_index()

    players_only_df

    # groupby player and season

    # loop through the dataframe and make sub dataframes that are grouped by season
    seasons = players_only_df['Season'].unique().tolist()

    # sort in reverse order
    seasons.sort(reverse=True)
    
    teams = players_only_df['Team'].unique().tolist()
    vs_teams = players_only_df['Opponent'].unique().tolist()
    ages = players_only_df['Age'].unique().tolist()
    nations = players_only_df['Nationality'].unique().tolist()
    positions = players_only_df['Position'].unique().tolist()
    referees = players_only_df['Referee'].unique().tolist()
    venues = players_only_df['Venue'].unique().tolist()
    players = players_only_df['Player'].unique().tolist()

    # create list of objects above
    categories = [seasons, teams, vs_teams, ages, nations, positions, referees, venues, players]

    # create a dictionary of dataframes, one for each season
    season_dfs = {}
    teams_dfs = {}
    vs_teams_dfs = {}
    ages_dfs = {}
    nations_dfs = {}
    positions_dfs = {}
    referees_dfs = {}
    venues_dfs = {}
    players_dfs = {}

    # for i in categories:
    #     if i:
    #         for j in i:
    #             if j:
    #                 for k in i:
    #                     if k:
    #                         season_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         teams_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         vs_teams_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         ages_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         nations_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         positions_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         referees_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]
    #                         venues_dfs[f"{j} vs {k}"] = players_only_df[(players_only_df['team'] == j) & (players_only_df['opponent'] == k)]



    for i in enumerate(categories):
        if i[1]:
            for j in i[1]:
                # create a dictionary as such but in a way thats iterable
                season_dfs[j] = players_only_df[players_only_df['Season'] == j]
                teams_dfs[j] = players_only_df[players_only_df['Team'] == j]
                vs_teams_dfs[j] = players_only_df[players_only_df['Opponent'] == j]
                ages_dfs[j] = players_only_df[players_only_df['Age'] == j]
                nations_dfs[j] = players_only_df[players_only_df['Nationality'] == j]
                positions_dfs[j] = players_only_df[players_only_df['Position'] == j]
                referees_dfs[j] = players_only_df[players_only_df['Referee'] == j]
                venues_dfs[j] = players_only_df[players_only_df['Venue'] == j]
                players_dfs[j] = players_only_df[players_only_df['Player'] == j]

    stat = 'Aerials Won Per90'
    x_cat = 'Player'
        
    for season in seasons:
        season_data = players_only_df[players_only_df['Season'] == season]

        # Calculate value counts for 'Nationality' and select top 200
        top_of_stat = season_data[x_cat].value_counts().head(200).index

        # Filter data for the top 20 x_cat for the given stat
        filtered_data = season_data[season_data[x_cat].isin(top_of_stat)]

        # filtered_data = season_data[season_data[x_cat].isin(top_of_stat)]

        # get the top 20 of each x that will be selected so that figure stays consistent
        
        # teams_dfs[team] = players_only_df[players_only_df['team'] == team]

        fig = px.scatter(
            filtered_data,
            x=x_cat,
            y=stat,
            color=stat,
            color_continuous_scale='reds',
            hover_data=['Player'],
            
        )

        st.info(f"Displaying {season} data for {stat} by {x_cat}")
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def dropdown_for_player_stats(players_only_df, selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5):

    # create a list of all the players
    players = players_only_df['Player'].unique().tolist()

    # create a dropdown menu for the players
    selected_player = st.sidebar.selectbox('Select Player', players, index=players.index(selected_player))

    # create a list of all seasons
    seasons = players_only_df['Season'].unique().tolist()

    # create a dropdown menu for the seasons
    selected_season = st.sidebar.selectbox('Select Season', seasons, index=seasons.index(selected_season))

    # create a list of all the stats from the numeric columns
    stats = players_only_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # create a dropdown menu for the stats
    selected_stat1 = st.sidebar.selectbox('Select Stat #1', stats, index=stats.index(selected_stat1))

    selected_stat2 = st.sidebar.selectbox('Select Stat #2', stats, index=stats.index(selected_stat2))

    selected_stat3 = st.sidebar.selectbox('Select Stat #3', stats, index=stats.index(selected_stat3))

    selected_stat4 = st.sidebar.selectbox('Select Stat #4', stats, index=stats.index(selected_stat4))

    selected_stat5 = st.sidebar.selectbox('Select Stat #5', stats, index=stats.index(selected_stat5))

    return selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5


def calculate_per90(df):
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

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

    return df_grouped

def normalize_df(df):
    """_summary_

    Args:
        df (_type_): _description_
    """

def process_player_datav2(players_only_df):

    selected_player = 'Bruno Fernandes'
    selected_season = '2022'
    selected_stat1 = 'Sca Per90'
    selected_stat2 = 'Pass xA Per90'
    selected_stat3 = 'Assists Per90'
    selected_stat4 = 'Gca Per90'
    selected_stat5 = 'Aerials Won Per90'

    print(players_only_df.columns.tolist())

    # create selectbox for player and stat
    selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = dropdown_for_player_stats(players_only_df, selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5)

    # filter for selected player
    player_df = players_only_df[(players_only_df['Player'] == selected_player) & (players_only_df['Season'] == selected_season)]

    # create a five point chart for the players top 5 per90 stats
    fig = px.line_polar(
        player_df,
        r=[player_df[selected_stat1].mean(), player_df[selected_stat2].mean(), player_df[selected_stat3].mean(), player_df[selected_stat4].mean(), player_df[selected_stat5].mean()],
        theta=[selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5],
        line_close=True,
        title=f"{selected_player}'s Top 5 Per90 Stats"
    )

    st.info(f"Displaying {selected_player}'s top 5 Per90 stats for {selected_season}")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

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

# def display_quant_stats(selected_teams_df, selected_team, selected_opponent):

#     def get_quant_stats_over_time(selected_teams_df, is_selected_team):
#         result_df = pd.DataFrame()

#         if is_selected_team:
#             result_df['Goals Scored'] = selected_teams_df['selected_team_score']
#             result_df['Goals Conceded'] = selected_teams_df['selected_opponent_score']
#             result_df['xG For'] = selected_teams_df['selected_team_xg']
#             result_df['xG Against'] = selected_teams_df['selected_opponent_xg']
#             result_df['Clean Sheets'] = (selected_teams_df['selected_opponent_score'] == 0).astype(int)
#         else:
#             result_df['Goals Scored'] = selected_teams_df['selected_opponent_score']
#             result_df['Goals Conceded'] = selected_teams_df['selected_team_score']
#             result_df['xG For'] = selected_teams_df['selected_opponent_xg']
#             result_df['xG Against'] = selected_teams_df['selected_team_xg']
#             result_df['Clean Sheets'] = (selected_teams_df['selected_team_score'] == 0).astype(int)

#         result_df['Date'] = selected_teams_df['date']
#         result_df.sort_values(by='Date', inplace=True)

#         return result_df

#     # Extract results into a list
#     def format_dataframe(df, team):
#         st.dataframe(
#             df,
#             column_config={
#                 "Team Results Over Time": st.column_config.LineChartColumn(
#                     f"{selected_team} Results Over Time",
#                     help="Results over time for " + team + " and their opponent. Draws are indicated by the null result (0).",
#                     y_min=-1,
#                     y_max=1,
#                 ),
#                 "Opponent Results Over Time": st.column_config.LineChartColumn(
#                     f"{selected_opponent} Results Over Time",
#                     help="Results over time for " + team + " and their opponent. Draws are indicated by the null result (0).",
#                     y_min=-1,
#                     y_max=1,
#                 ),
#             },
#         )

#     selected_team_df = get_quant_stats_over_time(selected_teams_df, True)
#     selected_opponent_df = get_quant_stats_over_time(selected_teams_df, False)

#     # transpose the dataframes
#     selected_team_df = selected_team_df.T
#     selected_opponent_df = selected_opponent_df.T

#     # create a list of the values for each each across all columns
#     selected_team_list = selected_team_df.values.tolist()
#     selected_opponent_list = selected_opponent_df.values.tolist()

#     # the list as a whole is a list of lists, create a dictionary with the column names as the keys
#     selected_team_dict = dict(zip(selected_team_df.index, selected_team_list))
#     selected_opponent_dict = dict(zip(selected_opponent_df.index, selected_opponent_list))

#     # now we can remove the date column from the dictionary
#     del selected_team_dict['Date']
#     del selected_opponent_dict['Date']

#     # convert

#     print(selected_team_list)
#     print(selected_opponent_list)

#     # now each list goes back into a dataframe with the stats as the index
#     selected_team_df = pd.DataFrame(selected_team_list, index=selected_team_df.index, columns=[selected_team])
#     selected_opponent_df = pd.DataFrame(selected_opponent_list, index=selected_opponent_df.index, columns=[selected_opponent])

#     print(selected_team_df)
#     print(selected_opponent_df)

#     # call the format_dataframe function for both the selected team and opponent dataframes
#     format_dataframe(selected_team_df, selected_team)
#     format_dataframe(selected_opponent_df, selected_opponent)

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

    st.info(f"**{selected_team}** stats over time:")
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


# def display_qual_stats(selected_teams_df, selected_team, selected_opponent):

#     def get_qual_stats_over_time(team):
#         # Determine the 'home' and 'away' conditions based on the selected team
#         home_condition = (selected_teams_df['home_team'] == team)
#         away_condition = (selected_teams_df['away_team'] == team)
        
#         # Define the conditions for a win, draw, or loss
#         win_condition = ((home_condition) & (selected_teams_df['home_score'] > selected_teams_df['away_score'])) | ((away_condition) & (selected_teams_df['home_score'] < selected_teams_df['away_score']))
#         draw_condition = ((home_condition | away_condition) & (selected_teams_df['home_score'] == selected_teams_df['away_score']))
#         loss_condition = ~win_condition & ~draw_condition

#         # For each match, assign the result (Win, Draw, Loss)
#         selected_teams_df.loc[win_condition, 'result'] = 'Win'
#         selected_teams_df.loc[draw_condition, 'result'] = 'Draw'
#         selected_teams_df.loc[loss_condition, 'result'] = 'Loss'

#         # Keep only the rows for the selected team and the new 'result' column
#         team_df = selected_teams_df.loc[home_condition | away_condition, ['date', 'result']]

#         # Convert the 'result' series to a list and store it in a DataFrame
#         results_list = team_df['result'].tolist()
#         results_df = pd.DataFrame({'Results Distribution Over Time': [results_list]}, index=[team])

#         return results_df

#     def format_special_qual_columns(team_df):
#         st.dataframe(
#             team_df,
#             column_config={
#                 "Results Over Time": st.column_config.ListColumn(
#                     "Results Over Time",
#                     width="medium",
#                     help="Win/Draw/Loss results over time",
#                 ),
#             },
#             use_container_width=True,
#         )

#     team_df = get_qual_stats_over_time(selected_team)
#     opponent_df = get_qual_stats_over_time(selected_opponent)

#     format_special_qual_columns(team_df)
#     format_special_qual_columns(opponent_df)

# # def get_qual_stats(selected_teams_df, selected_team, selected_opponent):

#     teams = [selected_team, selected_opponent]
    
#     qual_stats_df = pd.DataFrame()

#     # Get team and opponent stats
#     for team in teams:
#         # Determine the 'home' and 'away' conditions based on the current team
#         home_condition = (selected_teams_df['home_team'] == team)
#         away_condition = (selected_teams_df['away_team'] == team)
        
#         # Calculate the statistics for each team
#         total_games = selected_teams_df[home_condition | away_condition].shape[0]
        
#         wins_condition = ((home_condition) & (selected_teams_df['home_score'] > selected_teams_df['away_score'])) | ((away_condition) & (selected_teams_df['home_score'] < selected_teams_df['away_score']))
#         total_wins = selected_teams_df[wins_condition].shape[0]
        
#         draws_condition = ((home_condition | away_condition) & (selected_teams_df['home_score'] == selected_teams_df['away_score']))
#         total_draws = selected_teams_df[draws_condition].shape[0]
        
#         losses_condition = ~wins_condition & ~draws_condition
#         total_losses = selected_teams_df[losses_condition].shape[0]
        
#         # Add the statistics to the dataframe
#         qual_stats = pd.Series({'Total Games': total_games, 'Total Wins': total_wins, 'Total Losses': total_losses, 'Total Draws': total_draws}, name=team)
#         qual_stats_df = qual_stats_df.append(qual_stats)

#     # transpose the dataframe
#     qual_stats_df = qual_stats_df.T

#     # round the values
#     qual_stats_df = qual_stats_df.round(2)
    
#     return qual_stats_df

# def process_players_stats(selected_teams_df, selected_team, selected_opponent):

# display_quant_stats
# Load the data from the db
def main():
    
    # show the time the page was last  updated with timestamp
    st.write("Last updated: ", datetime.now())
    
    # Load the data from the db
    players_df, results_df = load_data_from_csv()

    # clean the data
    players_df, results_df = clean_data(players_df, results_df)

    # df_per90 = calculate_per90(players_df)
    # df_per90 = df_per90.reset_index(drop=True)   

    # players_only_df = rename_columns(df_per90)

    # # reorder the columns with player then season then rest of cols
    # first_cols = ['Player', 'Season']
    # rest_of_cols = [col for col in players_only_df.columns if col not in first_cols]
    # players_only_df = players_only_df[first_cols + rest_of_cols]

    # # selected_player, selected_season, selected_stat1, selected_stat2, selected_stat3, selected_stat4, selected_stat5 = dropdown_for_player_stats(players_only_df)

    # process_player_datav2(players_only_df)

    # process_player_datav2(players_df)
    # process_player_data(players_df)

    # Create a multiselect for the seasons
    selected_seasons, filtered_df = create_multiselect_seasons(players_df)

    # Create a multiselect for the teams
    selected_team, selected_opponent, filtered_df = create_dropdown_teams(
        filtered_df)
    
    # call filter_df_by_team_and_opponent
    selected_teams_df, grouped_player_df = filter_df_by_team_and_opponent(filtered_df, selected_team, selected_opponent)

    print(selected_teams_df.columns.tolist())

    # reorder columns so selected_team and selected_opponent are first
    selected_teams_df = selected_teams_df[['selected_team', 'selected_opponent', 'matchup_id', 'season_matchup_id', 'selected_team_score', 'selected_opponent_score', 'selected_team_draw', 'selected_team_won', 'selected_team_lost', 'selected_opponent_won', 'selected_opponent_lost', 'selected_team_xg', 'selected_opponent_xg', 'selected_team_result_category', 'selected_opponent_result_category', 'selected_team_normalized_result', 'selected_opponent_normalized_result', 'selected_team_home', 'selected_opponent_home', 'date', 'home_team', 'away_team']]

    print(selected_teams_df)

    display_qual_stats(selected_teams_df, selected_team, selected_opponent)

    display_quant_stats(selected_teams_df, selected_team, selected_opponent)

    # Get the results dataframe
    results_df = get_results_df(selected_teams_df, selected_team, selected_opponent)

    match_quick_facts(selected_teams_df, grouped_player_df, selected_team, selected_opponent)

    # selected_team_results = get_results_list(selected_teams_df, selected_team)
    # selected_opponent_results = get_results_list(selected_teams_df, selected_opponent)

    # # Apply color formatting to the results
    # selected_team_results_formatted = color_format(selected_team_results)
    # selected_opponent_results_formatted = color_format(selected_opponent_results)

    # Generate a time histogram
    plost.time_hist(
        data=results_df,
        date='date',
        x_unit='month',
        y_unit='year',
        color='team_result',
        aggregate='count',
        title=f'{selected_team} vs {selected_opponent} Results Distribution',
        legend='bottom'
    )

    # # Display the results in Streamlit
    # st.markdown(f"**{selected_team} vs {selected_opponent} Results Distribution:**")
    # st.markdown(selected_team_results_formatted, unsafe_allow_html=True)
    # st.markdown(selected_opponent_results_formatted, unsafe_allow_html=True)

    team_stats_og, team_stats_df = get_teams_stats(selected_teams_df, selected_team, selected_opponent)

    show_teams_stats_v2(team_stats_df)

    # qual_stats_df = get_qual_stats(selected_teams_df, selected_team, selected_opponent)

    # show_teams_stats(qual_stats_df, cm)

    # quant_stats_df = get_quant_stats(selected_teams_df, selected_team, selected_opponent)

    # show_teams_stats(quant_stats_df, cm)

    # player_stats_df = get_players_stats(player_level_df, selected_team, selected_opponent)

    # show_teams_stats_dataframe(team_stats_df)

    # show_players_stats_dataframe(player_stats_df)

    # show_team_stats_html(team_stats_df)

    # show_player_stats_html(player_stats_df)

    # display_dataframe(player_stats_df)

    # display_players_stats(player_stats)

    # call get_teams_stats()
    # stats_for_team = get_teams_stats(filtered_df, selected_team, selected_opponent)
    
    # # get team stats
    # if 'team_score' not in filtered_df.columns or 'opponent_score' not in filtered_df.columns:
    #     st.error("Dataframe doesn't contain 'team_score' or 'opponent_score'. Check your data cleaning and filtering process.")

    # stats_for_team, stats_for_opponent, team, opponent = get_teams_stats_v2(filtered_df, selected_team, selected_opponent)

    # Show the stats for the two teams selected
    # show_stats_for_teams(stats_for_team, stats_for_opponent, team, opponent)




# run the main function
if __name__ == "__main__":
    main()

