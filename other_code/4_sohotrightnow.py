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
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import unicodedata
import plotly.graph_objects as go
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join('./scripts')))

from constants import color1, color2, color3, color4, color5, cm, fbref_leagues as leagues, fbref_cats as cats, seasons, color_dark1, fbref_base_url

from files import big5_players_csv, data_out, scraped_big5_data

from functions import scraping_current_fbref, normalize_encoding

# make this directory if it doesn't exist
if not os.path.exists(scraped_big5_data):
    os.makedirs(scraped_big5_data)

# create full url for scraping
# season to scrape
season = [season for season in seasons if season == '2023-2024'][0]

gw1_df = scraping_current_fbref(cats)

print(gw1_df)