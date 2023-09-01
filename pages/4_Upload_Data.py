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

def add_construction():
    return st.info("""#### 🏗️ :orange[This app is under construction]""")

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
def get_avg_proj_pts(players, projections, status):
    
    average_proj_pts = 0

    for status in players['Status'].unique():
        top_10, reserves, top_10_proj_pts = filter_by_status_and_position(players, projections, status)

        print(f"Average projected points for {status} top 10 players: {top_10_proj_pts}")

        average_proj_pts += top_10_proj_pts

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
    print("Debug - Players after filtering:", available_players.head())
    
    # Filter the available players to remove players that are not in the "Waivers" status
    available_players = available_players[available_players['Status'] == 'Waivers']
    
    # Debug: Show filtered available_players
    print("Debug - available_players:", available_players.head())

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
    
    top_10_proj_pts = top_10['ProjFPts'].sum()

    return top_10, reserves, top_10_proj_pts

# Initialize session states
if 'only_starters' not in st.session_state:
    st.session_state.only_starters = False

if 'lineup_clicked' not in st.session_state:
    st.session_state.lineup_clicked = False

def main():
    # Adding construction banner or any other initial setups
    add_construction()

    mdlit(
    """### To get your optimal lineup head to -> @(https://www.fantrax.com/fantasy/league/d41pycnmlj3bmk8y/players;statusOrTeamFilter=ALL;pageNumber=1;positionOrGroup=SOCCER_NON_GOALIE;miscDisplayType=1) & follow the GIF below to populate the Players' data correctly and then download it
        """
        )   

    # Adding vertical space for better UI
    add_vertical_space(2)

    # Displaying GIF guide for users
    local_gif(fx_gif)  # Replace with the actual path to your GIF
    
    # File upload option for user
    uploaded_file = st.file_uploader("Upload a file", type="csv")

    if uploaded_file:
        # Reading the uploaded CSV file
        players = pd.read_csv(uploaded_file)
        # Assuming 'gw4_projections' is a file path or URL
        projections = load_csv(gw4_projections)  # Replace with the actual path

        print(f"Number of rows in all players: {len(players)}")
        
        debug_filtering(projections, players)

        print(f"Debug - Projections before filtering: {projections.head()}")
        print(f"Debug - Players before filtering: {players.head()}")
        # Checking if 'W (Thu)' is in the 'Status' column and renaming it if needed    

        # if value starts with "W " replace the entire value string with "Waivers"
        players['Status'] = players['Status'].str.replace(r'^W.*', 'Waivers', regex=True)

        unique_statuses = players['Status'].unique()

        print("Debug - unique_statuses:", unique_statuses)

        # get dataframe of players with status 'Waivers' or 'FA' to Waivers
        available_players = players[players['Status'].isin(['Waivers', 'FA'])]

        print("Debug - available_players:", available_players.head())

        print(f"Number of rows in available_players: {len(available_players)}")

        # Dropdown for user to select team
        st.write("### Select your Fantasy team from the dropdown below")
        status = st.selectbox('List of Teams', unique_statuses)

        with stylable_container(
            key="green_button",
            css_styles="""
                button {
                    background-color: #370617;
                    color: white;
                    border-radius: 20px;
                }
                """,
        ):

        
            # Button to process the selection
            if st.button('Get my optimal lineup') or st.session_state.lineup_clicked:
                st.session_state.lineup_clicked = True  # set this to True once the button is clicked

                # Displaying top 10 outfielders and reserves based on selected status
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    
                    # turn status to a list
                    status = [status]
                    
                    top_10, reserves, top_10_proj_pts = filter_by_status_and_position(players, projections, status)

                    # change status back to string
                    status_x = status[0]

                    st.write(f"### {status_x} Best XI")
                    st.dataframe(top_10, use_container_width=True)
                    st.write("### Reserves")
                    st.dataframe(reserves, use_container_width=True)

                with col2:
                    # Explicitly set the status to 'Waivers' here
                    status_waivers_fa = ['Waivers', 'FA']
                    
                                        # Debug print to check if 'available_players' is being fed into the function
                    print("Debug - Filtering by status_waivers:", status_waivers_fa)

                    # top_10_waivers, reserves_waivers, _ = filter_by_status_and_position(available_players, projections, status_waivers_fa)

                    status_waivers_fa_x = 'Waivers & FA'

                    st.write(f"### {status_waivers_fa_x} Best XI")

                    # Checkbox for "Only Starters?" that uses session_state
                    st.session_state.only_starters = st.checkbox('Only Starters?', value=st.session_state.only_starters)

                    if st.session_state.only_starters:
                        available_players_filtered = available_players[available_players['ProjGS'] == 1]
                    else:
                        available_players_filtered = available_players.copy()

                    # Get top 10 and reserves for waivers and FA
                    top_10_waivers, reserves_waivers, _ = filter_by_status_and_position(available_players_filtered, projections, status_waivers_fa)

                    # Debug print to check if 'top_10_waivers' and 'reserves_waivers' are empty
                    print("Debug - top_10_waivers:", top_10_waivers)
                    print("Debug - reserves_waivers:", reserves_waivers)

                    # top_10_waivers_filtered = dataframe_explorer(top_10_waivers)
                    st.dataframe(top_10_waivers, use_container_width=True)
                    st.write("### Reserves")
                    st.dataframe(reserves_waivers, use_container_width=True)

                    col1, col2 = st.columns(2)

                    with col1:

                        status = [status]

                        # call get_avg_proj_pts function to get the average projected points for top 10 players across all managers within the same positionitional limits
                        average_proj_pts = get_avg_proj_pts(players, projections, status)

                        status_x = status[0]

                        # show the total projected points for top 10 players as style_metric_cards
                        col1.metric(label="Total Projected FPts", value=top_10_proj_pts)

                        # with col2 we will show the average projected points across all managers top 10 players

                        col2.metric(label="Average Projected FPts of Best XIs across the Division", value=average_proj_pts, delta=top_10_proj_pts - average_proj_pts)

                        style_metric_cards()


if __name__ == "__main__":
    main()
