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

# function to load this csv /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/db_files/players.db and /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/csv_files/players.csv

file_path = 'data/data_out/final_data/csv_files/players.csv'
if os.path.exists(file_path):
    print(f"File {file_path} found.")
else:
    print(f"File {file_path} not found.")

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

    # Create a new column in the results_df called "team_won" that looks at team_score and opponent_score uaing np.where
    players_df["team_won"] = np.where(players_df["team_score"] > players_df["opponent_score"], 1, 0)

    # Create a new column in the results_df called "winning_xG"
    players_df["team_xG"] = np.where(team_is_home, players_df["home_xg"], players_df["away_xg"])

    # Create a new column in the results_df called "losing_xG"
    players_df["opponent_xG"] = np.where(team_is_home, players_df["away_xg"], players_df["home_xg"])

    # results_df
    print(f"Cleaned players_df columns:\n{players_df.columns.tolist()} ")

    # strip leading or trailing whitespace from all  columns
    players_df = players_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    return players_df, results_df

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
    """
    Summary:
        This function filters the df by the selected_team and selected_opponent.

    Args:
        df (pandas DataFrame): The df to be filtered.
        selected_team (str): The selected team.
        selected_opponent (str): The selected opponent.

    Returns:
        df (pandas DataFrame): The filtered df.
    """
    print(f"<> Running filter_df_by_team_and_opponent()...")

    # Filter the df by the selected_team and selected_opponent
    df = df[((df["team"] == selected_team) | (df["team"] == selected_opponent)) & 
        ((df["opponent"] == selected_team) | (df["opponent"] == selected_opponent))]

    print(f"<> Printing df after filtering by the selected_team and selected_opponent:\n   ---------------------------------------------------------------\n   == {df}")

    # print additional details this dataframe
    print(f"<> This dataframe is made up of the player and match statistics for {selected_team} vs {selected_opponent} from season {df['season_long'].unique()[0]} to {df['season_long'].unique()[-1]}.\n  ---------------------------------------------------------------\n<> Over that time {df['player'].nunique()} players have played in this fixture, with {df[df['team'] == selected_team]['player'].nunique()} players playing for {selected_team} and {df[df['team'] == selected_opponent]['player'].nunique()} players playing for {selected_opponent}.\n  ---------------------------------------------------------------\n{df['player'].value_counts().head(10)}")

    # print unique season_match_teams
    print(f"<> Printing unique season_match_teams:\n   ---------------------------------------------------------------\n   == {df['season_match_teams'].unique()}")

    # this is player level data that we can save for later
    # we can use this to create a dropdown for the user to select a player

    player_level_df = df.copy()

    # now we need to extract a single row for each match of the following columns
    # we can use season_match_teams and there should be 2 in each season 
    # then we want to create a new df of just the team level data and aggregate the following columns ['gameweek', 'season_long', 'season', 'match_teams', 'season_match_teams', 'home_team', 'away_team', 'home_xg', 'away_xg', 'home_score', 'away_score', 'date', 'referee', 'venue', 'dayofweek', 'start_time', 'attendance', 'winning_team', 'losing_team', 'season_gameweek_home_team_away_team', 'team_score', 'opponent_score', 'team_won', 'team_xG', 'opponent_xG'] 

    # create a new df of just the team level data
    team_level_df = df.groupby(['season_matchup_id', 'season_long', 'season', 'matchup_id', 'home_team', 'away_team', 'home_xg', 'away_xg', 'home_score', 'away_score', 'date', 'referee', 'venue', 'dayofweek', 'start_time', 'attendance', 'winning_team', 'losing_team', 'team_score', 'opponent_score', 'team_won', 'team_xG', 'opponent_xG', 'home']).agg({'team': 'first', 'opponent': 'first'}).reset_index()

    # print unique season_matchup_id
    print(f"<> Before dropping dupes... Printing nunique season_matchup_id:\n   ---------------------------------------------------------------\n   == {team_level_df['season_matchup_id'].nunique()}")

    # season_matchup_id is not unique as there are 2 rows for each match, one for each team, so we can drop duplicates of this column
    team_level_df.drop_duplicates(subset=['season_matchup_id'], inplace=True)

    # print unique season_matchup_id
    print(f"<> After dropping dupes... Printing nunique season_matchup_id:\n   ---------------------------------------------------------------\n   == {team_level_df['season_matchup_id'].nunique()}")

    # print this new df
    print(f"<> Printing team_level_df:\n   ---------------------------------------------------------------\n   == {team_level_df}")

    # season as string
    team_level_df['season'] = team_level_df['season'].astype(str)

    team_level_df['team_won'] = np.where(team_level_df['team'] == team_level_df['winning_team'], 1, 0)

    return team_level_df, player_level_df

def clean_player_level_df(df):
    """_summary_

    Args:
        df (_type_): _description_
    """

def print_df_columns(df):
    """_summary_

    Args:
        df (_type_): _description_
    """
    print(f"<> Printing df columns:\n   ---------------------------------------------------------------\n   == {df.columns.tolist()}")

def clean_team_level_df(df):
    
    df = df.copy()
    df['season'] = df['season'].astype(str)
    df = df[['team', 'home', 'opponent', 'team_score', 'opponent_score', 'home_team', 'away_team', 'home_xg', 'away_xg', 'winning_team', 'losing_team', 'team_xG', 'opponent_xG', 'date', 'referee', 'venue', 'dayofweek', 'start_time', 'attendance']]

    # strip "," and convert attendance to int
    df['attendance'] = df['attendance'].str.replace(',', '').fillna(0).astype(int)
    
    df = df.groupby(['team', 'home']).agg({
        'home_xg': ['mean', 'sum'], 
        'away_xg': ['mean', 'sum'], 
        'winning_team': lambda x: (x == x.name[0]).sum(),
        'losing_team': lambda x: (x == x.name[0]).sum(),
        'team_xG': ['mean', 'sum'],
        'opponent_xG': 'sum',
        'team_score': ['mean', 'sum'],
        'opponent_score': ['mean', 'sum'],
        'date': 'count', 
        'attendance': 'mean'
    }).reset_index()
    
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    df.rename(columns={
        'team_': 'Team', 
        'home_': 'Home',
        'home_xg_sum': 'Total xG Home',
        'away_xg_sum': 'Total xG Away',
        'home_xg_mean': 'Average Home xG',
        'away_xg_mean': 'Average Away xG',
        'winning_team_<lambda>': 'Total Wins',
        'losing_team_<lambda>': 'Total Losses',
        'team_xG_sum': 'Total xG For',
        'team_xG_mean': 'Average xG For',
        'opponent_xG_sum': 'Total xG Against',
        'team_score_sum': 'Total Goals Scored',
        'team_score_mean': 'Average Goals Scored',
        'opponent_score_sum': 'Total Goals Conceded',
        'opponent_score_mean': 'Average Goals Conceded',
        'date_count': 'Total Games',
        'attendance_mean': 'Average Attendance'
    }, inplace=True)

    df['Total Draws'] = df['Total Games'] - df['Total Wins'] - df['Total Losses']
    
    for loc in ['Home', 'Away']:
        df[f'Percent {loc} Games Won'] = df[f'Total Wins'] / df['Total Games'] * 100
        df[f'Percent {loc} Games Lost'] = df[f'Total Losses'] / df['Total Games'] * 100
        df[f'Percent {loc} Games Drawn'] = df['Total Draws'] / df['Total Games'] * 100

    # show df in app
    st.write(df)

    return df


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

    st.info("Matchup stats below...")

    st.dataframe(team_level_df, use_container_width=True)

    # get selected team stats
    selected_team_df = team_level_df[(team_level_df['team'] == selected_team) & (team_level_df['opponent'] == selected_opponent)]
    teams_stats = {
        'Total Games': len(selected_team_df),
        'Total Wins': len(selected_team_df[selected_team_df['winning_team'] == selected_team]),
        'Total Losses': len(selected_team_df[selected_team_df['losing_team'] == selected_team]),
        'Total Draws': len(selected_team_df[selected_team_df['winning_team'] == 'draw']),
        'Total Goals Scored': selected_team_df['team_score'].sum(),
        'Total Goals Conceded': selected_team_df['opponent_score'].sum(),
        'xG For': selected_team_df['team_xG'].sum(),
        'xG Against': selected_team_df['opponent_xG'].sum(),
        'Clean Sheets': len(selected_team_df[selected_team_df['opponent_score'] == 0])
    }

    # get selected opponent stats
    selected_opponent_df = team_level_df[(team_level_df['team'] == selected_opponent) & (team_level_df['opponent'] == selected_team)]
    opponent_stats = {
        'Total Games': len(selected_opponent_df),
        'Total Wins': len(selected_opponent_df[selected_opponent_df['winning_team'] == selected_opponent]),
        'Total Losses': len(selected_opponent_df[selected_opponent_df['losing_team'] == selected_opponent]),
        'Total Draws': len(selected_opponent_df[selected_opponent_df['winning_team'] == 'draw']),
        'Total Goals Scored': selected_opponent_df['team_score'].sum(),
        'Total Goals Conceded': selected_opponent_df['opponent_score'].sum(),
        'xG For': selected_opponent_df['team_xG'].sum(),
        'xG Against': selected_opponent_df['opponent_xG'].sum(),
        'Clean Sheets': len(selected_opponent_df[selected_opponent_df['opponent_score'] == 0])
    }

    # merge the two stats dicts
    matchup_stats_df = pd.DataFrame([teams_stats, opponent_stats], index=[selected_team, selected_opponent])

    # transpose the dataframe
    matchup_stats_df = matchup_stats_df.T

    st.dataframe(matchup_stats_df, use_container_width=True)

    return teams_stats, opponent_stats

def match_quick_facts(team_level_df, player_level_df, selected_team, selected_opponent):
    """Summary:
        This function returns the quick facts for the selected team and opponent using fstrings and vectorized computations after defining the following: `for team in teams:
            # Determine the 'home' and 'away' conditions based on the current team
            home_condition = (team_level_df['home_team'] == team)
            away_condition = (team_level_df['away_team'] == team)`.
        These stats are:
        - Total Games
        - Total Goals
        - Total players used
        - Top player by appearances
        - Teams' average attendance
         

    Args:
        df (_type_): _description_
        selected_team (_type_): _description_
        selected_opponent (_type_): _description_
    """

    st.info("Need to add matchup quick facts below...")
    """

    st.dataframe(player_level_df, use_container_width=True)

    st.info("Matchup quick facts below...")

    teams = [selected_team, selected_opponent]

    for team in teams:
        # Determine the 'home' and 'away' conditions based on the current team
        home_condition = (team_level_df['home_team'] == team)
        away_condition = (team_level_df['away_team'] == team)

        # Get the number of games played by the team
        total_games = team_level_df[home_condition | away_condition]['date'].count()

        # Get the number of goals scored by the team
        total_goals = team_level_df[home_condition | away_condition]['team_score'].sum()

        # Get the number of players used by the team
        total_players = player_level_df

        # Get the top player by appearances
        top_player = player_level_df[home_condition | away_condition].groupby('player')['player'].count().sort_values(ascending=False).index[0]

        # Get the average attendance
        average_attendance = team_level_df[home_condition | away_condition]['attendance'].mean()

        # Print the quick facts
        st.write(f"**{team}** played **{total_games}** games, scored **{total_goals}** goals, used **{total_players}** players, had **{average_attendance:.0f}** average attendance, and had **{top_player}** as their top player by appearances.")
        """

def get_teams_stats(team_level_df, selected_team, selected_opponent):
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

    st.info("Matchup stats below...")

    teams = [selected_team, selected_opponent]
    
    team_stats_df = pd.DataFrame()

    # Get team and opponent stats
    for team in teams:
        # Determine the 'home' and 'away' conditions based on the current team
        home_condition = (team_level_df['home_team'] == team)
        away_condition = (team_level_df['away_team'] == team)
        
        # Calculate the statistics for each team
        total_games = team_level_df[home_condition | away_condition].shape[0]
        total_wins = team_level_df[((home_condition) & (team_level_df['home_score'] > team_level_df['away_score'])) | 
                                   ((away_condition) & (team_level_df['home_score'] < team_level_df['away_score']))].shape[0]
        total_draws = team_level_df[((home_condition | away_condition) & 
                                     (team_level_df['home_score'] == team_level_df['away_score']))].shape[0]
        total_losses = total_games - total_wins - total_draws
        goals_scored = team_level_df[home_condition]['home_score'].sum() + team_level_df[away_condition]['away_score'].sum()
        goals_conceded = team_level_df[home_condition]['away_score'].sum() + team_level_df[away_condition]['home_score'].sum()
        xG_for = team_level_df[home_condition]['home_xg'].sum() + team_level_df[away_condition]['away_xg'].sum()
        xG_against = team_level_df[home_condition]['away_xg'].sum() + team_level_df[away_condition]['home_xg'].sum()
        clean_sheets = team_level_df[((home_condition) & (team_level_df['away_score'] == 0)) | 
                                     ((away_condition) & (team_level_df['home_score'] == 0))].shape[0]
        
        # Add the statistics to the dataframe
        team_stats = pd.Series({'Total Games': total_games, 'Total Wins': total_wins, 'Total Losses': total_losses, 
                                'Total Draws': total_draws, 'Total Goals Scored': goals_scored, 
                                'Total Goals Conceded': goals_conceded, 'xG For': xG_for, 'xG Against': xG_against, 
                                'Clean Sheets': clean_sheets}, name=team)
        team_stats_df = team_stats_df.append(team_stats)

            # transpose the dataframe
    team_stats_df = team_stats_df.T

    st.dataframe(team_stats_df, use_container_width=True)
    
    return team_stats_df

def get_players_stats(player_level_df, selected_team, selected_opponent):
    stats_categories = ['nationality', 'age', 'minutes', 'goals', 'assists', 'pens_made', 'pens_att', 'shots', 'shots_on_target', 'cards_yellow', 'cards_red', 'touches', 'tackles', 'interceptions', 'blocks', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'passes_completed', 'passes', 'passes_pct', 'progressive_passes', 'carries', 'progressive_carries', 'take_ons', 'take_ons_won', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocked_shots', 'blocked_passes', 'tackles_interceptions', 'clearances', 'errors', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries_distance', 'carries_progressive_distance', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

    selected_stats_categories = st.multiselect("Select stats categories", stats_categories, default=['gca', 'npxg', 'xg_assist', 'pass_xa'])

    # print_df_columns(player_level_df)

    cm = sns.diverging_palette(255, 149, s=0, l=1, as_cmap=True)
    cm2 = sns.diverging_palette(175, 35, s=60, l=85, as_cmap=True)

    for team in [selected_team, selected_opponent]:
        st.title(team)  # Set the title to the current team

        team_condition = (player_level_df['team'] == team)

        for stat_category in selected_stats_categories:
            # Get the stats for the current team
            team_player_stats = player_level_df[team_condition].groupby(['player', 'team'])[stat_category].agg(['sum', 'mean']).reset_index()

            st.subheader(stat_category)  # Set the subtitle to the current stat category

            # Display the stats of the top 5 unique players
            team_player_stats = team_player_stats.sort_values(by=['sum', 'mean'], ascending=False).head(5)

            # Apply color gradients
            st.dataframe(team_player_stats.style.
                         background_gradient(cmap=cm, subset=['sum'], vmin=team_player_stats['sum'].min(), vmax=team_player_stats['sum'].max()).
                         background_gradient(cmap=cm2, subset=['mean'], vmin=team_player_stats['mean'].min(), vmax=team_player_stats['mean'].max()))

def display_players_stats(players_stats):
    """Summary:
        This function displays each DataFrame in the players_stats dictionary 
        as a table in the Streamlit app.

    Args:
        players_stats (dict): The players stats dictionary returned by get_players_stats().

    """
    # Iterate over the dictionary
    for stat_category, df in players_stats.items():
        # Display the stat category as a header
        st.header(stat_category)

        # Display the DataFrame as a table
        st.dataframe(df)

def show_stats_for_teams(stats_for_team, stats_for_opponent, team, opponent):
    """Summary:
        This function shows the stats for the selected teams. These stats should be displayed in a table that has team and opponent as columns and the stats as rows
        
        Args:
            grouped_df (pandas DataFrame): The grouped dataframe.
            selected_teams (list): The selected teams.
            
        Returns:
            None
        """
    print("Received stats_for_team: ", stats_for_team)
    print("Received stats_for_opponent: ", stats_for_opponent)

    # Create a new dataframe
    df = pd.DataFrame(
        [stats_for_team, stats_for_opponent], index=[team, opponent])
    
    print(df)

    # Transpose the dataframe
    df = df.T

    # Create a new column "Win Percentage"
    df.loc["Win Percentage"] = df.loc["Total Wins"] / df.loc["Total Games"]

    # Create a new column "Loss Percentage"
    df.loc["Loss Percentage"] = df.loc["Total Losses"] / df.loc["Total Games"]

    # Create a new column "Goals Scored Per Game"
    df.loc["Goals Scored Per Game"] = df.loc["Total Goals Scored"] / \
        df.loc["Total Games"]
    
    # Create a new column "Goals Conceded Per Game"
    df.loc["Goals Conceded Per Game"] = df.loc["Total Goals Conceded"] / \
        df.loc["Total Games"]
    
    # Create a new column "xG For Per Game"
    df.loc["xG For Per Game"] = df.loc["xG For"] / \
        df.loc["Total Games"]

    # Create a new column "xG Against Per Game"
    df.loc["xG Against Per Game"] = df.loc["xG Against"] / \
        df.loc["Total Games"]
    
    # Create a new column "Clean Sheets Percentage"
    df.loc["Clean Sheets Percentage"] = df.loc["Clean Sheets"] / \
        df.loc["Total Games"]
    
    # Create a new column "xG Difference"
    df.loc["xG Difference"] = df.loc["xG For"] - \
        df.loc["xG Against"]
    
    # Create a new column "xG Difference Per Game"
    df.loc["xG Difference Per Game"] = df.loc["xG Difference"] / \
        df.loc["Total Games"]
    
    # Create a new column "Clean Sheets Total" as integer
    df.loc["Clean Sheets Total"] = df.loc["Clean Sheets"].astype(int)

    # Create a new column "Clean Sheets Percentage"
    df.loc["Clean Sheets Percentage"] = df.loc["Clean Sheets"] / \
        df.loc["Total Games"]
    
    # format the column names so that they replace _ with a space
    df.columns = df.columns.str.replace("_", " ")

    # format the dataframe to 2 decimal places, the percentages as percentages and the rest as integers
    df = df.round(2)
    df.loc["Win Percentage"] = df.loc["Win Percentage"].apply(
        lambda x: "{:.2%}".format(x))
    df.loc["Loss Percentage"] = df.loc["Loss Percentage"].apply(
        lambda x: "{:.2%}".format(x))
    df.loc["Clean Sheets Percentage"] = df.loc["Clean Sheets Percentage"].apply(
        lambda x: "{:.2%}".format(x))
    
    # show the dataframe
    st.dataframe(df)

    
    players_df_filtered = players_df[(players_df['season'].isin(selected_seasons)) & ((players_df['team'] == selected_team) | (players_df['team'] == selected_opponent))]

    players_df_filtered = players_df_filtered.groupby(['player_name', 'team']).sum().reset_index()

    # make sure numerical columns are rounded to 2 decimals

    """

    for index, row in df_filtered.iterrows():
        if row['home_team'] == team1:
            stats_team1['Total Games'] += 1
            stats_team1['Total Goals Scored'] += row['home_score']
            stats_team1['Total Goals Conceded'] += row['away_score']
            stats_team1['xG For'] += row['home_xg']
            stats_team1['xG Against'] += row['away_xg']
            stats_team1['Clean Sheets'] += 1 if row['away_score'] == 0 else 0
            if row['winning_team'] == team1:
                stats_team1['Total Wins'] += 1
            elif row['losing_team'] == team1:
                stats_team1['Total Losses'] += 1

        if row['away_team'] == team1:
            stats_team1['Total Games'] += 1
            stats_team1['Total Goals Scored'] += row['away_score']
            stats_team1['Total Goals Conceded'] += row['home_score']
            stats_team1['xG For'] += row['away_xg']
            stats_team1['xG Against'] += row['home_xg']
            stats_team1['Clean Sheets'] += 1 if row['home_score'] == 0 else 0
            if row['winning_team'] == team1:
                stats_team1['Total Wins'] += 1
            elif row['losing_team'] == team1:
                stats_team1['Total Losses'] += 1

        if row['home_team'] == team2:
            stats_team2['Total Games'] += 1
            stats_team2['Total Goals Scored'] += row['home_score']
            stats_team2['Total Goals Conceded'] += row['away_score']
            stats_team2['xG For'] += row['home_xg']
            stats_team2['xG Against'] += row['away_xg']
            stats_team2['Clean Sheets'] += 1 if row['away_score'] == 0 else 0
            if row['winning_team'] == team2:
                stats_team2['Total Wins'] += 1
            elif row['losing_team'] == team2:
                stats_team2['Total Losses'] += 1

        if row['away_team'] == team2:
            stats_team2['Total Games'] += 1
            stats_team2['Total Goals Scored'] += row['away_score']
            stats_team2['Total Goals Conceded'] += row['home_score']
            stats_team2['xG For'] += row['away_xg']
            stats_team2['xG Against'] += row['home_xg']
            stats_team2['Clean Sheets'] += 1 if row['home_score'] == 0 else 0
            if row['winning_team'] == team2:
                stats_team2['Total Wins'] += 1
            elif row['losing_team'] == team2:
                stats_team2['Total Losses'] += 1

    return stats_team1, stats_team2

    """

def display_dataframe(df):
    # Convert the dataframe to HTML
    df_html = df.to_html(index=False)

    # Create a CSS style string
    style = """
    <style>
    table {
        line-height: 1.6;
        letter-spacing: 0.01em;
    }
    table.dataframe {
        font-family: 'Fira Code';
    }
    table.dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    table.dataframe th {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """

    # Create the final HTML string
    html = style + df_html

    # Display the HTML in Streamlit
    st.markdown(html, unsafe_allow_html=True)

# Load the data from the db
def main():
    
    # show the time the page was last  updated with timestamp

    st.write("Last updated: ", datetime.now())
    
    # Load the data from the db
    
    players_df, results_df = load_data_from_csv()

    # clean the data
    players_df, results_df = clean_data(players_df, results_df)

    # Create a multiselect for the seasons
    _, filtered_df = create_multiselect_seasons(players_df)

    # Create a multiselect for the teams
    selected_team, selected_opponent, filtered_df = create_dropdown_teams(
        filtered_df)
    
    # call filter_df_by_team_and_opponent
    team_level_df, player_level_df = filter_df_by_team_and_opponent(filtered_df, selected_team, selected_opponent)

    # call clean_team_level_df
    # df = clean_team_level_df(team_level_df)

    # call match_quick_facts
    # match_quick_facts(team_level_df, player_level_df, selected_team, selected_opponent)

    team_stats = get_teams_stats(team_level_df, selected_team, selected_opponent)

    player_stats = get_players_stats(player_level_df, selected_team, selected_opponent)

    display_dataframe(team_stats)

    display_dataframe(player_stats)

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

