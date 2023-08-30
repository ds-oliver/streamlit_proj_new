import streamlit as st
import pandas as pd
import os
import sys
from warnings import filterwarnings

filterwarnings('ignore')

st.set_page_config(
    page_title="Footy Magic",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

from files import gw4_projections  # Add other imports
from functions import load_csv  # Add other imports

def filter_by_status_and_position(players, projections):
    # Filter by status
    status = st.selectbox('Status', players['Status'].unique())
    filtered_players = players[players['Status'] == status]
    
    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]
    
    # Position based filtering
    pos_limit = {'D': 5, 'M': 5, 'F': 3}
    final_list = []
    
    for pos, limit in pos_limit.items():
        pos_players = projections[projections['Pos'] == pos]
        selected_players = pos_players.nlargest(limit, 'ProjFPts')
        final_list.append(selected_players)
    
    result = pd.concat(final_list).reset_index(drop=True)
    result = result.nlargest(10, 'ProjFPts').reset_index(drop=True)
    
    # Divide into top 10 outfielders and reserves
    top_10 = result.head(10)
    reserves = result.tail(len(result) - 10)
    
    return top_10, reserves

def main():
    uploaded_file = st.file_uploader("Upload a file", type="csv")

    if uploaded_file:
        players = pd.read_csv(uploaded_file)
        projections = load_csv(gw4_projections)  # Assuming load_csv is a function that reads the csv file into a DataFrame
    
        if st.button('Get data'):
            top_10, reserves = filter_by_status_and_position(players, projections)
            st.write("### Top 10 Outfielders")
            st.write(top_10)
            st.write("### Reserves")
            st.write(reserves)

if __name__ == "__main__":
    main()
