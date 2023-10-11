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
    layout="wide"
)

load_css()

filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

def apply_styles(box_shadow, background_color, border_size_px, border_color, border_radius_px, border_left_color):
    # Determine box shadow string based on the box_shadow argument
    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    # Inject CSS via a markdown block
    st.markdown(
        f"""
        <style>
            div[data-testid="metric-container"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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

                defenders = projections[projections['Position'] == 'D'].nlargest(d, 'ProjFPts')
                midfielders = projections[projections['Position'] == 'M'].nlargest(m, 'ProjFPts')
                forwards = projections[projections['Position'] == 'F'].nlargest(f, 'ProjFPts')

                current_combination = pd.concat([defenders, midfielders, forwards])
                current_score = current_combination['ProjFPts'].sum()

                if current_score > best_score:
                    best_combination = current_combination
                    best_score = current_score

    print(f"Total Defenders: {len(best_combination[best_combination['Position'] == 'D'])}")
    print(f"Total Midfielders: {len(best_combination[best_combination['Position'] == 'M'])}")
    print(f"Total Forwards: {len(best_combination[best_combination['Position'] == 'F'])}")

    # Sort DataFrame by 'Pos' in the order 'D', 'M', 'F' and then by 'ProjFPts'
    best_combination.sort_values(by=['Position', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Position' else x, ascending=[True, False], inplace=True)
    best_combination.reset_index(drop=True, inplace=True)

    reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)
    best_score = round(best_score, 1)

    # Drop the 'Priority' column before returning
    best_combination.drop(columns=['Priority'], inplace=True, errors='ignore')
    reserves.drop(columns=['Priority'], inplace=True, errors='ignore')

    return best_combination, reserves, best_score, projections

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

                defenders = projections[projections['Position'] == 'D'].nlargest(d, 'ProjFPts')
                midfielders = projections[projections['Position'] == 'M'].nlargest(m, 'ProjFPts')
                forwards = projections[projections['Position'] == 'F'].nlargest(f, 'ProjFPts')

                current_combination = pd.concat([defenders, midfielders, forwards])
                current_score = current_combination['ProjFPts'].sum()

                if current_score > best_score:
                    best_combination = current_combination
                    best_score = current_score

    print(f"Total Defenders: {len(best_combination[best_combination['Position'] == 'D'])}")
    print(f"Total Midfielders: {len(best_combination[best_combination['Position'] == 'M'])}")
    print(f"Total Forwards: {len(best_combination[best_combination['Position'] == 'F'])}")

    # Sort DataFrame by 'Pos' in the order 'D', 'M', 'F' and then by 'ProjFPts'
    best_combination.sort_values(by=['Position', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Position' else x, ascending=[True, False], inplace=True)
    best_combination.reset_index(drop=True, inplace=True)

    reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)

    # Drop the 'Priority' column before returning
    best_combination.drop(columns=['Priority'], inplace=True, errors='ignore')
    reserves.drop(columns=['Priority'], inplace=True, errors='ignore')

    return best_combination, reserves

# Function that will get the average projected points for top 10 players across all managers within the same positional limits
def get_avg_proj_pts(players, projections):
    total_proj_pts = 0
    num_statuses = len(players['Status'].unique())

    for status in players['Status'].unique():
        top_10, _, top_10_proj_pts, _ = filter_by_status_and_position(players, projections, status)
        total_proj_pts += top_10_proj_pts

    average_proj_pts = round((total_proj_pts / num_statuses), 1)
    return average_proj_pts

def get_filtered_players(players, projections, status, projgs_value=None):
    """
    Filter players based on their status and optionally their ProjGS values.
    
    Args:
    - players (pd.DataFrame): DataFrame containing player information.
    - projections (pd.DataFrame): DataFrame containing projections for players.
    - status (str or list): Player status values to filter by.
    - projgs_value (int, optional): ProjGS value to filter players by. Defaults to None.
    
    Returns:
    - pd.DataFrame: Filtered player projections.
    - pd.DataFrame: Filtered player information.
    """
    # Check for the presence of 'Status' column and return early if not found.
    if 'Status' not in players.columns:
        print("Warning: 'Status' column is not present in the players dataframe.")
        return pd.DataFrame(), pd.DataFrame()

    # Convert status to list if it's a string
    if isinstance(status, str):
        status = [status]

    # Filter players by their status
    filtered_players = players[players['Status'].isin(status)]

    # Optionally filter players by their ProjGS values
    if projgs_value is not None:
        filtered_players = filtered_players[filtered_players['ProjGS'] == projgs_value]

    if filtered_players.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Get a list of filtered player names
    player_list = filtered_players['Player'].unique().tolist()

    # Filter the projections DataFrame using the player list
    filtered_projections = projections[projections['Player'].isin(player_list)]

    # Prioritize players based on their ProjGS values
    filtered_projections['Priority'] = filtered_projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
    filtered_projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

    return filtered_projections, filtered_players

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

    uploaded_file = st.file_uploader("Upload your player data", type="csv")

    if uploaded_file:
        center_running()
        with st.spinner('Loading data...'):
            players = pd.read_csv(uploaded_file)
            projections = load_csv(proj_csv)
            ros_ranks_data = load_csv(ros_ranks)

            # Renaming columns for consistency
            players.rename(columns={'Pos': 'Position'}, inplace=True, errors='ignore')
            projections.rename(columns={'Pos': 'Position'}, inplace=True, errors='ignore')

            # Merging data
            projections = pd.merge(projections, ros_ranks_data, how='left', on='Player', suffixes=('', '_y'))
            projections.drop(columns=projections.filter(regex='_y'), inplace=True)

            # Dropping unnecessary columns
            for df in [players, projections]:
                df.drop(columns=[col for col in ['Pos', '+/-'] if col in df.columns], inplace=True)

            projections['ROS Rank'].fillna(200, inplace=True)
            
            debug_filtering(projections, players)

            players['Status'] = players['Status'].apply(lambda x: 'Waivers' if x.startswith('W (') else x)
            unique_statuses = sorted(players['Status'].unique())
            available_players = players[players['Status'].isin(['Waivers', 'FA'])]

            col_a, col_b = st.columns(2)

            with col_a:
                st.write("### 🛡️ Select your Fantasy team")
                status = st.selectbox('', unique_statuses)

            with col_b:
                st.session_state.only_starters = st.checkbox('Only consider starters?')

            if st.button('🚀 Get my optimal lineup') or st.session_state.lineup_clicked:
                st.session_state.lineup_clicked = True
                
                col1, col2 = st.columns(2)

                with col1:
                    status_list = [status]
                    top_10, reserves, top_10_proj_pts, roster = filter_by_status_and_position(players, projections, status_list)
                    st.write(f"### 🥇 {status} Best XI")
                    st.dataframe(top_10)
                    st.write("### 🔄 Reserves")
                    st.dataframe(reserves)

                with col2:
                    available_players = pd.merge(available_players, projections[['Player', 'ProjGS', 'ROS Rank']], on='Player', how='left')
                    top_10_waivers, reserves_waivers = filter_available_players_by_projgs(
                        available_players, projections, ['Waivers', 'FA'], 1 if st.session_state.only_starters else None
                    )
                    st.write("### 🚀 Waivers & FA Best XI")
                    st.dataframe(top_10_waivers)
                    st.write("### 🔄 Reserves")
                    st.dataframe(reserves_waivers)

                average_proj_pts = get_avg_proj_pts(players, projections)

                col_c, col_d = st.columns(2)

                with col_c:
                    
                    avg_ros_of_top_fas = available_players.sort_values(by=['ROS Rank'], ascending=True).head(5)['ROS Rank'].mean()
                
                    with st.expander("Performance Metrics"):
                        average_proj_pts = get_avg_proj_pts(players, projections)
                        average_ros_rank_of_roster = round(roster['ROS Rank'].mean(), 1)
                        # take difference between average ROS rank of roster and average ROS rank of top 5 FAs so that the higher the difference, the better the roster
                        ros_rank_diff = round(average_ros_rank_of_roster - avg_ros_of_top_fas, 1)
                        print(f"Average ROS Rank of roster: {average_ros_rank_of_roster}")
                        print(f"Average ROS Rank of top 5 FAs: {avg_ros_of_top_fas}")
                        print(f"ROS Rank difference: {ros_rank_diff}")

                        value_score = round((200 - average_ros_rank_of_roster) * top_10_proj_pts, 0)
                        # normalize the value score by dividing by the average projected points of the best XIs across the division
                        value_score = value_score / average_proj_pts

                        # work in ros_rank_diff into the value score
                        value_score = value_score + ros_rank_diff

                        print(f"Value score: {value_score}")
                        st.metric(label="🔥 Total Projected FPts", value=top_10_proj_pts)
                        st.metric(label="🌟 Average XI ROS Rank", value=round(top_10['ROS Rank'].mean(), 1))
                        st.metric(label="📊 Value Score", value=round((200 - average_ros_rank_of_roster) * top_10_proj_pts, 1))
                        st.metric(
                            label="💹 Average Projected FPts of Best XIs across the Division",
                            value=average_proj_pts, delta=round((top_10_proj_pts - average_proj_pts), 1)
                        )

                        value_score_df = pd.DataFrame(columns=['Status', 'Value Score'])
                        for status in players['Status'].unique():
                            top_10, _, top_10_proj_pts, _ = filter_by_status_and_position(players, projections, status)
                            average_ros_rank_of_roster = round(top_10['ROS Rank'].mean(), 1)
                            value_score = round((200 - average_ros_rank_of_roster) * top_10_proj_pts, 1)
                            value_score_df.loc[len(value_score_df)] = [status, value_score]
                        value_score_df.sort_values(by=['Value Score'], ascending=False, inplace=True)
                        value_score_df['Roster Rank'] = value_score_df['Value Score'].rank(method='dense', ascending=False).astype(int)

                with col_d:
                    with st.expander("📈 Value Score Rankings"):
                        st.dataframe(value_score_df)

            st.divider()

            if st.button('🔍 View all Projections'):
                projections = load_csv(proj_csv)
                st.dataframe(projections, use_container_width=True)

# def main():
#     # Adding construction banner or any other initial setups
#     add_construction()

#     custom_cmap = create_custom_sequential_cmap(*colors)

#     mdlit(
#     """### To get your optimal lineup head to -> @(https://www.fantrax.com/fantasy/league/d41pycnmlj3bmk8y/players;statusOrTeamFilter=ALL;pageNumber=1;positionOrGroup=SOCCER_NON_GOALIE;miscDisplayType=1) & follow the GIF below to populate and download the Players' data.
#         """
#         )

#     add_vertical_space(2)
#     local_gif(fx_gif)

#     uploaded_file = st.file_uploader("Upload a file", type="csv")

#     if uploaded_file:
#         center_running()
#         with st.spinner('Loading data...'):
#             players = pd.read_csv(uploaded_file)
#             projections = load_csv(proj_csv)
#             ros_ranks_data = load_csv(ros_ranks)

#             if 'Pos' in players.columns:
#                 players.rename(columns={'Pos': 'Position'}, inplace=True)

#             if 'Pos' in projections.columns:
#                 projections.rename(columns={'Pos': 'Position'}, inplace=True)

#             projections = pd.merge(projections, ros_ranks_data, how='left', on='Player', suffixes=('', '_y'))
#             projections = projections[projections.columns.drop(list(projections.filter(regex='_y')))]

#             cols_to_drop = ['Pos', '+/-']
#             projections = projections.drop([col for col in projections.columns if col in cols_to_drop], axis=1)
#             players = players.drop([col for col in players.columns if col in cols_to_drop], axis=1)
#             projections['ROS Rank'].fillna(200, inplace=True)

#             debug_filtering(projections, players)

#             players['Status'] = players['Status'].apply(lambda x: 'Waivers' if x.startswith('W (') else x)
#             unique_statuses = players['Status'].unique()
#             available_players = players[players['Status'].isin(['Waivers', 'FA'])]
#             print("Unique statuses in available players:", unique_statuses)

#             col_a, col_b = st.columns(2)

#             with col_a:
#                 st.write("### Select your Fantasy team from the dropdown below")
#                 status = st.selectbox('List of Teams', unique_statuses)
#                 with stylable_container(key="green_button", css_styles="..."):
#                     lineup_button = st.button('Get my optimal lineup')

#             with col_b:
#                 st.session_state.only_starters = st.checkbox('Only Starters?', value=st.session_state.only_starters)

#             if lineup_button or st.session_state.lineup_clicked:
#                 center_running()
#                 with st.spinner('Getting your optimal lineup...'):
#                     st.session_state.lineup_clicked = True
#                     st.divider()

#                     col1, col2 = st.columns(2)
#                     status_list = [status]

#                     # Use get_filtered_players for filtering by team status
#                     filtered_players, filtered_projections = get_filtered_players(players, projections, status_list)
#                     top_10, reserves, top_10_proj_pts = filter_by_status_and_position(filtered_players, filtered_projections, status_list)

#                     with col1:
#                         st.write(f"### {status} Best XI")
#                         st.dataframe(top_10)
#                         st.write("### Reserves")
#                         st.dataframe(reserves)

#                     with col2:
#                         projgs_value = 1 if st.session_state.only_starters else None
#                         filtered_available_players, filtered_projections = get_filtered_players(available_players, projections, ['Waivers', 'FA'], projgs_value)
                    
#                         # Log to check if the filtered players dataframe contains the 'Status' column
#                         print(f"Debug - filtered_available_players columns: {filtered_available_players.columns}")

#                         top_10_waivers, reserves_waivers = filter_by_status_and_position(filtered_available_players, filtered_projections, ['Waivers', 'FA'])

#                         st.write("### Waivers & FA Best XI")
#                         st.dataframe(top_10_waivers)
#                         st.write("### Reserves")
#                         st.dataframe(reserves_waivers)

#                     col1, col2 = st.columns(2)

#                     with col1:
#                         average_proj_pts = get_avg_proj_pts(players, projections)
#                         col1.metric(label="Total Projected FPts", value=top_10_proj_pts)
#                         average_ros_rank = round(top_10['ROS Rank'].mean(), 1)
#                         col1.metric(label="Starting XI ROS Rank Average", value=average_ros_rank)
#                         value_score = round((200 - average_ros_rank) * top_10_proj_pts, 1)
#                         col1.metric(label="Starting XI Value Score", value=value_score)

#                         value_score_df = pd.DataFrame(columns=['Status', 'Value Score'])
#                         for status in players['Status'].unique():
#                             top_10, _, top_10_proj_pts = filter_by_status_and_position(players, projections, status)
#                             average_ros_rank = round(top_10['ROS Rank'].mean(), 1)
#                             value_score = round((200 - average_ros_rank) * top_10_proj_pts, 1)
#                             value_score_df.loc[len(value_score_df)] = [status, value_score]
#                         value_score_df.sort_values(by=['Value Score'], ascending=False, inplace=True)
#                         value_score_df['Roster Rank'] = value_score_df['Value Score'].rank(method='dense', ascending=False).astype(int)

#                     with col2:
#                         col2.metric(label="Average Projected FPts of Best XIs across the Division", value=average_proj_pts, delta=round((top_10_proj_pts - average_proj_pts), 1))

#                     style_metric_cards(background_color=colors[0], border_color=colors[1], border_size_px=2)
#                     st.divider()
#                     value_score_df.drop(columns=['Value Score'], inplace=True)
#                     value_score_df.reset_index(drop=True, inplace=True)
#                     st.dataframe(value_score_df, use_container_width=True)

#     st.divider()
#     if st.button('View all Projections'):
#         projections = load_csv(proj_csv)
#         st.dataframe(projections, use_container_width=True)


if __name__ == "__main__":
    # # Initialize session states
    # if 'only_starters' not in st.session_state:
    #     st.session_state.only_starters = False

    # if 'lineup_clicked' not in st.session_state:
    #     st.session_state.lineup_clicked = False

    main()
