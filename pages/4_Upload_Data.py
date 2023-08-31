import streamlit as st
import pandas as pd
import os
import sys
from warnings import filterwarnings
import base64

filterwarnings('ignore')

st.set_page_config(
    page_title="Footy Magic",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

from files import gw4_projections, fx_gif
from functions import load_csv, add_construction

def local_gif(file_path):
    with open(file_path, "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    return st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="download data">',
        unsafe_allow_html=True,
    )

def filter_by_status_and_position(players, projections, status):
    filtered_players = players[players['Status'] == status]
    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]

    pos_limit = {'D': 5, 'M': 5, 'F': 3}
    final_list = []

    for pos, limit in pos_limit.items():
        pos_players = projections[projections['Pos'] == pos]
        selected_players = pos_players.nlargest(limit, 'ProjFPts')
        final_list.append(selected_players)

    top_10 = pd.concat(final_list).nlargest(10, 'ProjFPts')
    top_10.sort_values(by='Pos', key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}), inplace=True)
    
    reserves = projections.drop(top_10.index).reset_index(drop=True)

    return top_10, reserves

def main():
    add_construction()

    local_gif(fx_gif)
    
    uploaded_file = st.file_uploader("Upload a file", type="csv")

    if uploaded_file:
        players = pd.read_csv(uploaded_file)
        projections = load_csv(gw4_projections)
        
        unique_statuses = [status for status in players['Status'].unique() if status != 'W (Thu)']
        status = st.selectbox('Status', unique_statuses)

        if st.button('Get data'):
            st.write("### Top 10 Outfielders and Reserves based on the Selected Status")
            
            top_10, reserves = filter_by_status_and_position(players, projections, status)
            st.write("### Top 10 Outfielders")
            st.write(top_10)
            st.write("### Reserves")
            st.write(reserves)

if __name__ == "__main__":
    main()
