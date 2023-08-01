import streamlit as st
import pandas as pd
import numpy as np
# 1_load_data_from_dbt.py
import sys
import os

# Add the scripts directory to the Python path
sys.path.append(
    '/Users/hogan/dev/streamlit_proj_new/scripts/')

from chunk_and_save_data import load_data_from_db

# Load the data from the db
players_table, results_table = load_data_from_db()

st.write(results_table)

# def to filter the dbs into smaller tables
def filter_dbs(db, table, column, value):
    """
    This function filters the db into a smaller table.
    """
    # Filter the db into a smaller table
    filtered_table = db[db[column] == value]

    return filtered_table

# Filter the players table into a smaller table
players_table_filtered = filter_dbs(players_table, "players", "player", 1)

st.write(players_table_filtered)