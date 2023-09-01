import streamlit as st
import pandas as pd
import os
import sys
from warnings import filterwarnings
import base64
import streamlit_extras
from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.customize_running import center_running

from files import gw4_projections, fx_gif
from functions import load_csv, add_construction, load_css

st.set_page_config(
    page_title="Footy Magic",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)


def local_gif(file_path):
    with open(file_path, "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    return st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="download data" width="100%">',
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

    mdlit(
    """-> To get your optimal lineup navigate to [Fantrax league home](https://www.fantrax.com/fantasy/league/d41pycnmlj3bmk8y/players;statusOrTeamFilter=ALL;pageNumber=1;positionOrGroup=SOCCER_NON_GOALIE;miscDisplayType=1)...
        Follow the GIF below to populate the Players' data correctly <-
        """
        )   
    
    add_vertical_space(2)

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
