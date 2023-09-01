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

def filter_by_status_and_position(players, projections, status):
    # Filter players by status
    filtered_players = players[players['Status'] == status]
    player_list = filtered_players['Player'].unique().tolist()

    # Filter projections by the list of players with the selected status
    projections = projections[projections['Player'].isin(player_list)]

    # Positional limits for 'D': Defense, 'M': Midfield, 'F': Forward
    pos_limit = {'D': 5, 'M': 5, 'F': 3}

    # Get the top players by position based on their projected fantasy points
    final_list = [
        projections[projections['Pos'] == pos].nlargest(limit, 'ProjFPts')
        for pos, limit in pos_limit.items()
    ]

    # Get the top 10 players across all positions
    top_10 = pd.concat(final_list).nlargest(10, 'ProjFPts')

    # Sort by position and reset the index
    top_10.sort_values(by='Pos', key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}), inplace=True)
    top_10.reset_index(drop=True, inplace=True)

    # Get the remaining players as reserves
    # Here I reset the index before dropping
    projections.reset_index(drop=True, inplace=True)
    top_10.reset_index(drop=True, inplace=True)

    # get total projected points for top 10 players 
    top_10_proj_pts = top_10['ProjFPts'].sum()

    # Using isin() to match players rather than index
    reserves = (projections[~projections['Player'].isin(top_10['Player'])].reset_index(drop=True)).head(5)

    return top_10, reserves, top_10_proj_pts

# function that will get the average projected points for top 10 players across all managers within the same positional limits
def get_avg_proj_pts(players, projections, status):
    
    average_proj_pts = 0

    for status in players['Status'].unique():
        top_10, reserves, top_10_proj_pts = filter_by_status_and_position(players, projections, status)

        print(f"Average projected points for {status} top 10 players: {top_10_proj_pts}")

        average_proj_pts += top_10_proj_pts

    return average_proj_pts


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

        #rename 'W (Thu)' values in Status column to 'Waivers'
        players['Status'] = players['Status'].replace({'W (Thu)': 'Waivers'})

        # Extracting unique statuses except 'W (Thu)'
        unique_statuses = [status for status in players['Status'].unique() if status != 'Waivers']

        # capitalizing the first letter of each unique_statuses
        unique_statuses = [status.capitalize() for status in unique_statuses]

        # get dataframe of players with status 'W (Thu)'
        available_players = players[players['Status'] == 'W (Thu)']

        # drop Status column
        available_players.drop(columns=['Status'], inplace=True)

        # Dropdown for user to select team
        st.write("### Select your Fantasy team from the dropdown below")
        status = st.selectbox('List of Teams', unique_statuses)

        # lowercase the status
        status = status.lower()

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
            if st.button('Get my optimal lineup'):
                # Displaying top 10 outfielders and reserves based on selected status
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    
                    top_10, reserves, top_10_proj_pts = filter_by_status_and_position(players, projections, status)
                    st.write(f"### {status} Best XI")
                    st.dataframe(top_10, use_container_width=True)
                    st.write("### Reserves")
                    st.dataframe(reserves, use_container_width=True)

                with col2:
                    
                    status = 'Waivers'
                    
                    # call filter_by_status_and_position function to get top 10 players and reserves
                    top_10, reserves, top_10_proj_pts = filter_by_status_and_position(players, projections, 'Waivers')

                    st.write(f"### {status} Best XI")
                    st.dataframe(top_10, use_container_width=True)
                    st.write("### Reserves")
                    st.dataframe(reserves, use_container_width=True)

                    col1, col2 = st.columns(2)

                    with col1:

                        # call get_avg_proj_pts function to get the average projected points for top 10 players across all managers within the same positional limits
                        average_proj_pts = get_avg_proj_pts(players, projections, status)

                        # show the total projected points for top 10 players as style_metric_cards
                        col1.metric(label="Total Projected FPts", value=top_10_proj_pts)

                        # with col2 we will show the average projected points across all managers top 10 players

                        col2.metric(label="Average Projected FPts of Best XIs across the Division", value=average_proj_pts, delta=top_10_proj_pts - average_proj_pts)

                        style_metric_cards()




if __name__ == "__main__":
    main()
