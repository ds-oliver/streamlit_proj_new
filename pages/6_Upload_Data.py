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

from files import gw4_projections, fx_gif  # Add other imports
from functions import load_csv  # Add other imports

# retrieve local gif file
def local_gif(file_path):
    file_ = open(file_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
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

    result = pd.concat(final_list).reset_index(drop=True)
    result = result.nlargest(10, 'ProjFPts').reset_index(drop=True)
    result.sort_values(by='Pos', key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}), inplace=True)

    top_10 = result.head(10)
    reserves = pd.concat([result, top_10]).drop_duplicates(keep=False).reset_index(drop=True)
    
    return top_10, reserves

def main():
    uploaded_file = st.file_uploader("Upload a file", type="csv")

    # call local gif file
    local_gif(fx_gif)

    if uploaded_file:
        players = pd.read_csv(uploaded_file)
        projections = load_csv(gw4_projections)  # Assuming load_csv is a function that reads the csv file into a DataFrame

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
