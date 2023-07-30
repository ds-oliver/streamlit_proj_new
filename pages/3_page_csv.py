import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging
import sqlite3

# function to load this csv /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/db_files/players.db and /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/csv_files/players.csv

file_path = 'data/data_out/final_data/csv_files/players.csv'
if os.path.exists(file_path):
    print(f"File {file_path} found.")
else:
    print(f"File {file_path} not found.")

@st.cache
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

# @st.cache
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

def clean_data(players_df, results_df):
    """
    This function cleans the data.
    """
    print("Cleaning data...")

    print(players_df.columns.tolist())
    print(results_df.columns.tolist())

    # merge ['season', 'gameweek', 'home_team', 'away_team'] columns from results_df to players_df
    players_df = players_df.merge(results_df[['season', 'gameweek', 'home_team', 'away_team']], how='left', on=[
                                    'season', 'gameweek', 'home_team', 'away_team'])
    
    # drop ['home_team', 'away_team'] columns from players_df

    # create team column from home_team and opponent from away_team
    players_df["team"] = players_df["winning_team"]
    players_df["opponent"] = players_df["losing_team"]

    results_df["team"] = results_df["winning_team"]
    results_df["opponent"] = results_df["losing_team"]

    # convert season to int
    results_df["season"] = results_df["season"].astype(int)
    players_df["season"] = players_df["season"].astype(int)

    # Create a new column in the results_df called "winning_score"
    results_df["winning_score"] = results_df.apply(
        lambda x: x["score"] if x["score"] > x["opponent_score"] else x["opponent_score"], axis=1)

    # Create a new column in the results_df called "losing_score"
    results_df["losing_score"] = results_df.apply(
        lambda x: x["score"] if x["score"] < x["opponent_score"] else x["opponent_score"], axis=1)

    # Create a new column in the results_df called "winning_xG"
    results_df["winning_xG"] = results_df.apply(
        lambda x: x["xG"] if x["score"] > x["opponent_score"] else x["xGA"], axis=1)

    # Create a new column in the results_df called "losing_xG"
    results_df["losing_xG"] = results_df.apply(
        lambda x: x["xG"] if x["score"] < x["opponent_score"] else x["xGA"], axis=1)
    
    return players_df, results_df


def create_multiselect_seasons(results_df):
    """
    Summary:
        This function creates a multiselect for the seasons.

    Args:
        results_df (pandas DataFrame): The results df.

        
    Returns:
        selected_seasons (list): The selected seasons.
    """
    # Get all unique seasons from the dataframe
    seasons = sorted(results_df["season"].unique())

    # Create a multiselect for the seasons
    selected_seasons = st.multiselect(
        "Select Season(s)", seasons, key="seasons")

    return selected_seasons


def create_dropdown_teams(results_df, selected_seasons):
    """
    Summary:
        This function creates a dropdown for the teams. Based on the selected_seasons list, this function creates a dropdown for the teams from those seasons.

    Args:
        results_df (pandas DataFrame): The results df.
        selected_seasons (list): The selected seasons.

    Returns:
        selected_teams (str): The selected team.
    """
    # Filter the DataFrame based on selected seasons
    if selected_seasons:  # Only if there are any selected seasons
        filtered_df = results_df[results_df["season"].isin(selected_seasons)]
        # Get all unique teams from the filtered dataframe
        teams = sorted(filtered_df["team"].unique())
    else:
        teams = []

    # Create a dropdown for the teams
    # st.info("Select two teams to compare.\nSelect first team:")

    selected_team = st.selectbox(
        "Select Team", teams, key="teams")
    
    # st.info("Select opponent:")

    opponents = [team for team in teams if team != selected_team]

    selected_opponent = st.selectbox(
        "Select Opponent", opponents, key="opponents")

    return selected_team, selected_opponent, filtered_df

def get_teams_stats(df, team, opponent):
    stats_for_team = {
        'total_games': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_goals_scored': 0,
        'total_goals_conceded': 0,
        'xG For': 0,
        'xG Against': 0,
        'Clean Sheets': 0
    }
    stats_for_opponent = {
        'total_games': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_goals_scored': 0,
        'total_goals_conceded': 0,
        'xG For': 0,
        'xG Against': 0,
        'Clean Sheets': 0
    }

    df_filtered = df[(df['team'] == team) | (df['team'] == opponent)]

    for index, row in df_filtered.iterrows():
        if row['team'] == team:
            stats_for_team['total_games'] += 1
            stats_for_team['total_goals_scored'] += row['score']
            stats_for_team['total_goals_conceded'] += row['opponent_score']
            stats_for_team['xG For'] += row['xG']
            stats_for_team['xG Against'] += row['xGA']
            stats_for_team['Clean Sheets'] += 1 if row['opponent_score'] == 0 else 0
            if row['winning_team'] == team:
                stats_for_team['total_wins'] += 1
            elif row['losing_team'] == team:
                stats_for_team['total_losses'] += 1

        if row['team'] == opponent:
            stats_for_opponent['total_games'] += 1
            stats_for_opponent['total_goals_scored'] += row['opponent_score']
            stats_for_opponent['total_goals_conceded'] += row['score']
            stats_for_opponent['xG For'] += row['xGA']
            stats_for_opponent['xG Against'] += row['xG']
            stats_for_opponent['Clean Sheets'] += 1 if row['score'] == 0 else 0
            if row['winning_team'] == opponent:
                stats_for_opponent['total_wins'] += 1
            elif row['losing_team'] == opponent:
                stats_for_opponent['total_losses'] += 1

    return stats_for_team, stats_for_opponent

def get_players_stats(players_df, selected_seasons ,selected_team, selected_opponent):
    
    players_df_filtered = players_df[(players_df['season'].isin(selected_seasons)) & ((players_df['team'] == selected_team) | (players_df['team'] == selected_opponent))]

    players_df_filtered = players_df_filtered.groupby(['player_name', 'team']).sum().reset_index()

    # make sure numerical columns are rounded to 2 decimals

    """

    for index, row in df_filtered.iterrows():
        if row['home_team'] == team1:
            stats_team1['total_games'] += 1
            stats_team1['total_goals_scored'] += row['home_score']
            stats_team1['total_goals_conceded'] += row['away_score']
            stats_team1['xG For'] += row['home_xg']
            stats_team1['xG Against'] += row['away_xg']
            stats_team1['Clean Sheets'] += 1 if row['away_score'] == 0 else 0
            if row['winning_team'] == team1:
                stats_team1['total_wins'] += 1
            elif row['losing_team'] == team1:
                stats_team1['total_losses'] += 1

        if row['away_team'] == team1:
            stats_team1['total_games'] += 1
            stats_team1['total_goals_scored'] += row['away_score']
            stats_team1['total_goals_conceded'] += row['home_score']
            stats_team1['xG For'] += row['away_xg']
            stats_team1['xG Against'] += row['home_xg']
            stats_team1['Clean Sheets'] += 1 if row['home_score'] == 0 else 0
            if row['winning_team'] == team1:
                stats_team1['total_wins'] += 1
            elif row['losing_team'] == team1:
                stats_team1['total_losses'] += 1

        if row['home_team'] == team2:
            stats_team2['total_games'] += 1
            stats_team2['total_goals_scored'] += row['home_score']
            stats_team2['total_goals_conceded'] += row['away_score']
            stats_team2['xG For'] += row['home_xg']
            stats_team2['xG Against'] += row['away_xg']
            stats_team2['Clean Sheets'] += 1 if row['away_score'] == 0 else 0
            if row['winning_team'] == team2:
                stats_team2['total_wins'] += 1
            elif row['losing_team'] == team2:
                stats_team2['total_losses'] += 1

        if row['away_team'] == team2:
            stats_team2['total_games'] += 1
            stats_team2['total_goals_scored'] += row['away_score']
            stats_team2['total_goals_conceded'] += row['home_score']
            stats_team2['xG For'] += row['away_xg']
            stats_team2['xG Against'] += row['home_xg']
            stats_team2['Clean Sheets'] += 1 if row['home_score'] == 0 else 0
            if row['winning_team'] == team2:
                stats_team2['total_wins'] += 1
            elif row['losing_team'] == team2:
                stats_team2['total_losses'] += 1

    return stats_team1, stats_team2

    """

def prepare_df_for_streamlit(filtered_df):
    """
    Summary:
        This function prepares the df for streamlit, the goals is to show the statistics for the two teams selected.
    
    Args:
        filtered_df (_type_): _description_
        selected_teams (_type_): _description_

    Returns:
        _type_: _description_
    """
    # group by team and aggregate all of the stats by sum
    grouped_df = filtered_df.groupby("team").sum()

    return grouped_df

def show_stats_for_teams(grouped_df, selected_teams):
    """
    Summary:
        This function shows the stats for the two teams selected.
    
    Args:
        grouped_df (_type_): _description_
        selected_teams (_type_): _description_

    Returns:
        _type_: _description_
    """
    # show the stats for the two teams selected
    st.write(grouped_df)    

def select_season(db, season):
    """
    This function selects the season from the db.
    """
    # Select the season from the db using sqlite3
    cur = db.cursor()
    cur.execute(
        "SELECT * FROM results WHERE season = ?", (season,))
    results = cur.fetchall()

    return results

def st_write_df(df):
    """
    This function writes the df to the streamlit app.
    """
    st.write(df)

def st_write_db(db):
    """
    This function writes the db to the streamlit app.
    """
    st.write(db)



# Load the data from the db
def main():
    
    # Load the data from the db
    
    players_df, results_df = load_data_from_csv()

    # clean the data
    players_df, results_df = clean_data(players_df, results_df)

    # Create a multiselect for the seasons
    selected_seasons = create_multiselect_seasons(results_df)

    # Create a multiselect for the teams
    selected_team, selected_opponent, filtered_df = create_dropdown_teams(
        results_df, selected_seasons)

    # Prepare the df for streamlit
    grouped_df = prepare_df_for_streamlit(filtered_df)

    # Show the stats for the two teams selected
    show_stats_for_teams(grouped_df, selected_team)

    # Select the season from the db using sqlite3


    # Write the df to the streamlit app
    st_write_df(grouped_df)


# run the main function
if __name__ == "__main__":
    main()

