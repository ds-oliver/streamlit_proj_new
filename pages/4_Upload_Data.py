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

from files import projections, fx_gif
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

# function that will get the average projected points for top 10 players across all managers within the same positionitional limits
def get_avg_proj_pts(players, projections):
    total_proj_pts = 0
    num_statuses = len(players['Status'].unique())

    for status in players['Status'].unique():
        top_10, _, top_10_proj_pts = filter_by_status_and_position(players, projections, status)
        print(f"Average projected points for {status} top 10 players: {top_10_proj_pts}")
        total_proj_pts += top_10_proj_pts

    average_proj_pts = round((total_proj_pts / num_statuses), 1)
    return average_proj_pts


def debug_filtering(projections, players):
    # Ensure that the data frames are not empty
    if projections.empty or players.empty:
        print("Debug - One or both DataFrames are empty.")
        return
    
    print("Debug - Projections before filtering:", projections.head())
    print("Debug - Players before filtering:", players.head())

    # Filter the projections DataFrame
    projections_filtered = projections[projections['ProjFPts'] >= 10]
    
    # Debug: Show filtered projections
    print("Debug - Projections after filtering:", projections_filtered.head())

    # Filter the players DataFrame to keep only those in the filtered projections
    available_players = players[players['Player'].isin(projections_filtered['Player'])]
    
    # Debug: Show filtered players
    print("Debug - available_players after filtering:", available_players.head())
    
    waivers_fa = ['Waivers', 'FA']


    # Filter the available players to remove players that are not in the "Waivers" or "FA" status
    filtered_available_players = players[players['Status'].isin(waivers_fa)]
    
    # Debug: Show filtered available_players
    print("Debug - available_players:", filtered_available_players.head())

def filter_by_status_and_position(players, projections, status):
    print(f"Debug - Filtering by status: {status}")

    if isinstance(status, str):
        status = [status]
    
    # Filter players by status
    filtered_players = players[players['Status'].isin(status)]
    print(f"Debug - Filtered Players count: {len(filtered_players)}")
    
    if filtered_players.empty:
        print("Debug - No players found for this status.")
        return pd.DataFrame(), pd.DataFrame(), 0
    
    # Filter projections based on the players list
    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]
    print(f"Debug - Projections count after filtering: {len(projections)}")
    
    if projections.empty:
        print("Debug - No projections found for these players.")
        return pd.DataFrame(), pd.DataFrame(), 0

    pos_limit = {'D': 5, 'M': 5, 'F': 3}
    final_list = [
        projections[projections['Pos'] == pos].nlargest(limit, 'ProjFPts')
        for pos, limit in pos_limit.items()
    ]
    
    top_10 = pd.concat(final_list).nlargest(10, 'ProjFPts')
    top_10.sort_values(by='Pos', key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}), inplace=True)
    top_10.reset_index(drop=True, inplace=True)
    
    projections.reset_index(drop=True, inplace=True)
    reserves = projections[~projections['Player'].isin(top_10['Player'])].head(5).reset_index(drop=True)
    
    top_10_proj_pts = round((top_10['ProjFPts'].sum()), 1)

    return top_10, reserves, top_10_proj_pts

def filter_available_players_by_projgs(players, projections, status, projgs_value):
    print(f"Debug - Filtering by status: {status} and ProjGS: {projgs_value}")
    
    # Ensure 'status' is a list
    if isinstance(status, str):
        status = [status]
    
    # Filter players by status
    filtered_players = players[players['Status'].isin(status)]
    print(f"Debug - Filtered Players count: {len(filtered_players)}")
    
    if filtered_players.empty:
        print("Debug - No players found for this status.")
        return pd.DataFrame(), pd.DataFrame()

    # Further filter by ProjGS value if specified
    if projgs_value is not None:
        filtered_players = filtered_players[filtered_players['ProjGS'] == projgs_value]
        print(f"Debug - Further filtered by ProjGS to {len(filtered_players)} players")
    
    # Filter projections based on the filtered players list
    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]
    print(f"Debug - Projections count after filtering: {len(projections)}")
    
    if projections.empty:
        print("Debug - No projections found for these players.")
        return pd.DataFrame(), pd.DataFrame()

    pos_limit = {'D': 5, 'M': 5, 'F': 3}
    final_list = [
        projections[projections['Pos'] == pos].nlargest(limit, 'ProjFPts')
        for pos, limit in pos_limit.items()
    ]
    
    top_10 = pd.concat(final_list).nlargest(10, 'ProjFPts')
    top_10.sort_values(by='Pos', key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}), inplace=True)
    top_10.reset_index(drop=True, inplace=True)
    
    reserves = projections[~projections['Player'].isin(top_10['Player'])].head(5).reset_index(drop=True)

    return top_10, reserves

# Initialize session states
if 'only_starters' not in st.session_state:
    st.session_state.only_starters = False

if 'lineup_clicked' not in st.session_state:
    st.session_state.lineup_clicked = False

def main():
    # Adding construction banner or any other initial setups
    add_construction()

    mdlit(
    """### To get your optimal lineup head to -> @(https://www.fantrax.com/fantasy/league/d41pycnmlj3bmk8y/players;statusOrTeamFilter=ALL;pageNumber=1;positionOrGroup=SOCCER_NON_GOALIE;miscDisplayType=1) & follow the GIF below to populate and download the Players' data.
        """
        )   

    add_vertical_space(2)
    local_gif(fx_gif)

    uploaded_file = st.file_uploader("Upload a file", type="csv")

    if uploaded_file:
        center_running()
        with st.spinner('Loading data...'):
            players = pd.read_csv(uploaded_file)
            projections = load_csv(projections)
            
            debug_filtering(projections, players)

            players['Status'] = players['Status'].str.replace(r'^W.*', 'Waivers', regex=True)
            unique_statuses = players['Status'].unique()
            available_players = players[players['Status'].isin(['Waivers', 'FA'])]

            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("### Select your Fantasy team from the dropdown below")
                status = st.selectbox('List of Teams', unique_statuses)
                with stylable_container(key="green_button", css_styles="..."):
                    lineup_button = st.button('Get my optimal lineup')
            
            with col_b:
                st.session_state.only_starters = st.checkbox('Only Starters?', value=st.session_state.only_starters)

            if lineup_button or st.session_state.lineup_clicked:
                center_running()
                with st.spinner('Getting your optimal lineup...'):
                    st.session_state.lineup_clicked = True
                    st.divider()
                    
                    col1, col2 = st.columns(2)

                    with col1:
                        status_list = [status]
                        top_10, reserves, top_10_proj_pts = filter_by_status_and_position(players, projections, status_list)
                        st.write(f"### {status} Best XI")
                        st.dataframe(top_10)
                        st.write("### Reserves")
                        st.dataframe(reserves)

                    with col2:
                        available_players = pd.merge(available_players, projections[['Player', 'ProjGS']], on='Player', how='left')
                        
                        if st.session_state.only_starters:
                            top_10_waivers, reserves_waivers = filter_available_players_by_projgs(available_players, projections, ['Waivers', 'FA'], 1)
                        else:
                            top_10_waivers, reserves_waivers = filter_available_players_by_projgs(available_players, projections, ['Waivers', 'FA'], None)
                        
                        st.write("### Waivers & FA Best XI")
                        st.dataframe(top_10_waivers)
                        st.write("### Reserves")
                        st.dataframe(reserves_waivers)

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        average_proj_pts = get_avg_proj_pts(players, projections)
                        col1.metric(label="Total Projected FPts", value=top_10_proj_pts)

                    with col2:
                        col2.metric(label="Average Projected FPts of Best XIs across the Division", value=average_proj_pts, delta=round((top_10_proj_pts - average_proj_pts), 1))

                    style_metric_cards()


if __name__ == "__main__":
    main()
