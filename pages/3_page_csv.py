import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging
import sqlite3

# function to load this csv /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/db_files/players.db and /Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/csv_files/players.csv

@st.cache
def load_data_from_db():
    """
    This function loads data from two tables in a SQLite database.
    Returns two pandas DataFrames, one for each table.
    """
    # Create a connection to the database
    conn = sqlite3.connect(
        '/Users/hogan/dev/streamlit_proj_new/data/mydatabase.db')

    # Load data from the 'players' table into a DataFrame
    players_table = pd.read_sql_query("SELECT * from players", conn)

    # Load data from the 'results' table into a DataFrame
    results_table = pd.read_sql_query("SELECT * from results", conn)

    # Don't forget to close the connection when you're done
    conn.close()

    return players_table, results_table

@st.cache
def load_data_from_csv():
    """
    This function loads the data from the csvs.
    """
    # Load the data from the csv

    players_df = pd.read_csv(
        "/Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/csv_files/players.csv")
    results_df = pd.read_csv(   
        "/Users/hogan/dev/streamlit_proj_new/data/data_out/final_data/csv_files/results.csv")
    
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
    # Create a multiselect for the seasons
    seasons = results_df["season"].unique()
    selected_seasons = st.multiselect(
        "Select Season(s)", seasons, default=seasons)

    return selected_seasons

def create_multiselect_teams(results_df, selected_seasons):
    """
    Summary:
        This function creates a multiselect for the teams. Based on the selected_seasons list, this function creates a multiselect for the teams from those seasons. 

    Args:
        results_df (pandas DataFrame): The results df.
        selected_seasons (list): The selected seasons.

    Returns:
        selected_teams (list): The selected teams.
    """
    # filter by seasons selected
    filtered_df = results_df[results_df["season"].isin(selected_seasons)]
    # Create a multiselect for the teams
    teams = filtered_df["team"].unique()

    # only allow two teams to be selected
    selected_teams = st.multiselect(
        "Select Team(s)", teams, default=teams)

    return selected_teams, filtered_df

def prepare_df_for_streamlit(filtered_df, selected_teams):
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

    # Create a multiselect for the seasons
    selected_seasons = create_multiselect_seasons(results_df)

    # Create a multiselect for the teams
    selected_teams, filtered_df = create_multiselect_teams(results_df, selected_seasons)

    # Prepare the df for streamlit
    grouped_df = prepare_df_for_streamlit(filtered_df, selected_teams)

    # Show the stats for the two teams selected
    show_stats_for_teams(grouped_df, selected_teams)

    # Select the season from the db using sqlite3


    # Write the df to the streamlit app
    st_write_df(grouped_df)


if __name__ == "__main__":
    main()


