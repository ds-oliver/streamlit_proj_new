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

from functions import * 

# load player data
players_data, _ = load_data_from_csv()

# clean player data
players_data, _ = clean_data(players_data, _)

# process player data
players_data, season_dfs, teams_dfs, vs_teams_dfs, ages_dfs, nations_dfs, positions_dfs, referees_dfs, venues_dfs = process_player_data(players_data)