import streamlit as st
import pandas as pd
import numpy as np
from chunk_and_save_data import load_data_from_db

data = load_data_from_db()

st.write(data)