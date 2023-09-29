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

from constants import colors
from files import projections as proj_csv, fx_gif, ros_ranks
from functions import load_csv, add_construction, load_css, create_custom_sequential_cmap

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
# def get_avg_proj_pts(players, projections):
#     total_proj_pts = 0
#     num_statuses = len(players['Status'].unique())

#     for status in players['Status'].unique():
#         top_10, _, top_10_proj_pts = filter_by_status_and_position(players, projections, status)
#         print(f"Average projected points for {status} top 10 players: {top_10_proj_pts}")
#         total_proj_pts += top_10_proj_pts

#     average_proj_pts = round((total_proj_pts / num_statuses), 1)
#     return average_proj_pts

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
    if isinstance(status, str):
        status = [status]

    filtered_players = players[players['Status'].isin(status)]

    if filtered_players.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]

    if projections.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    # Prioritize players with ProjGS not equal to 0
    projections['Priority'] = projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
    projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

    pos_limits = {'D': (3, 5), 'M': (2, 5), 'F': (1, 3)}
    max_players = 10
    best_combination = None
    best_score = 0

    for d in range(pos_limits['D'][0], pos_limits['D'][1] + 1):
        for m in range(pos_limits['M'][0], pos_limits['M'][1] + 1):
            for f in range(pos_limits['F'][0], pos_limits['F'][1] + 1):
                if d + m + f != max_players:
                    continue

                defenders = projections[projections['Pos'] == 'D'].nlargest(d, 'ProjFPts')
                midfielders = projections[projections['Pos'] == 'M'].nlargest(m, 'ProjFPts')
                forwards = projections[projections['Pos'] == 'F'].nlargest(f, 'ProjFPts')

                current_combination = pd.concat([defenders, midfielders, forwards])
                current_score = current_combination['ProjFPts'].sum()

                if current_score > best_score:
                    best_combination = current_combination
                    best_score = current_score

    print(f"Total Defenders: {len(best_combination[best_combination['Pos'] == 'D'])}")
    print(f"Total Midfielders: {len(best_combination[best_combination['Pos'] == 'M'])}")
    print(f"Total Forwards: {len(best_combination[best_combination['Pos'] == 'F'])}")

    # Sort DataFrame by 'Pos' in the order 'D', 'M', 'F' and then by 'ProjFPts'
    best_combination.sort_values(by=['Pos', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Pos' else x, ascending=[True, False], inplace=True)
    best_combination.reset_index(drop=True, inplace=True)

    reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)
    best_score = round(best_score, 1)

    return best_combination, reserves, best_score


# Function that will get the average projected points for top 10 players across all managers within the same positional limits
def get_avg_proj_pts(players, projections):
    total_proj_pts = 0
    num_statuses = len(players['Status'].unique())

    for status in players['Status'].unique():
        top_10, _, top_10_proj_pts = filter_by_status_and_position(players, projections, status)
        total_proj_pts += top_10_proj_pts

    average_proj_pts = round((total_proj_pts / num_statuses), 1)
    return average_proj_pts


# Filter available players by their ProjGS and status
def filter_available_players_by_projgs(players, projections, status, projgs_value):
    if isinstance(status, str):
        status = [status]

    filtered_players = players[players['Status'].isin(status)]
    
    if projgs_value is not None:
        filtered_players = filtered_players[filtered_players['ProjGS'] == projgs_value]

    if filtered_players.empty:
        return pd.DataFrame(), pd.DataFrame()

    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]

    if projections.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Prioritize players with ProjGS not equal to 0
    projections['Priority'] = projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
    projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

    pos_limits = {'D': (3, 5), 'M': (2, 5), 'F': (1, 3)}
    max_players = 10
    best_combination = None
    best_score = 0

    for d in range(pos_limits['D'][0], pos_limits['D'][1] + 1):
        for m in range(pos_limits['M'][0], pos_limits['M'][1] + 1):
            for f in range(pos_limits['F'][0], pos_limits['F'][1] + 1):
                if d + m + f != max_players:
                    continue

                defenders = projections[projections['Pos'] == 'D'].nlargest(d, 'ProjFPts')
                midfielders = projections[projections['Pos'] == 'M'].nlargest(m, 'ProjFPts')
                forwards = projections[projections['Pos'] == 'F'].nlargest(f, 'ProjFPts')

                current_combination = pd.concat([defenders, midfielders, forwards])
                current_score = current_combination['ProjFPts'].sum()

                if current_score > best_score:
                    best_combination = current_combination
                    best_score = current_score

    print(f"Total Defenders: {len(best_combination[best_combination['Pos'] == 'D'])}")
    print(f"Total Midfielders: {len(best_combination[best_combination['Pos'] == 'M'])}")
    print(f"Total Forwards: {len(best_combination[best_combination['Pos'] == 'F'])}")

    # Sort DataFrame by 'Pos' in the order 'D', 'M', 'F' and then by 'ProjFPts'
    best_combination.sort_values(by=['Pos', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Pos' else x, ascending=[True, False], inplace=True)
    best_combination.reset_index(drop=True, inplace=True)

    reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)

    return best_combination, reserves


# Initialize session states
if 'only_starters' not in st.session_state:
    st.session_state.only_starters = False

if 'lineup_clicked' not in st.session_state:
    st.session_state.lineup_clicked = False

def main():
    # Adding construction banner or any other initial setups
    add_construction()

    custom_cmap = create_custom_sequential_cmap(*colors)

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
            projections = load_csv(proj_csv)
            ros_ranks_data = load_csv(ros_ranks)

            projections = pd.merge(projections, ros_ranks, how='left', on='Player')
            
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

    # divider 
    st.divider()

    # add a button to "View all Projections" which will show the projections DataFrame
    if st.button('View all Projections'):
        projections = load_csv(proj_csv)

        st.dataframe(projections, use_container_width=True)


if __name__ == "__main__":
    main()
