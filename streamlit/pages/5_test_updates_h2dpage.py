import streamlit as st
import pandas as pd
import numpy as np
import re
import uuid

# @st.cache_data
def load_player_data():
    player_df = pd.read_csv('data/all_seasons_combined_df_2023-07-25_12-50-09.csv')
    player_df = player_df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('None'))
    drop_cols = ['Unnamed: 0', 'shirtnumber']
    player_df = player_df.drop(drop_cols, axis=1)
    return player_df

def process_player_data(player_df):
    player_df['year'] = player_df['season'].str[:4]
    player_df = player_df.rename(columns={'season': 'season_long', 'year': 'season', 'position_1': 'position'})

    # create a per90 column for minutes which is minutes / 90
    player_df['minutes_per90'] = player_df['minutes'] / 90
    
    # create match_teams column from team and opponent where we sort the team and opponent alphabetically and join with _
    player_df['match_teams'] = player_df.apply(lambda row: '_'.join(sorted([row['team'], row['opponent']])), axis=1)

    # create season_match_teams column from match_teams and season
    player_df['season_match_teams'] = player_df['match_teams'] + '_' + player_df['season'].astype(str)

    conditions = [
        (player_df['home'] == True),
        (player_df['home'] == False)
    ]

    choices_team = [player_df['team'], player_df['opponent']]
    choices_opponent = [player_df['opponent'], player_df['team']]

    player_df['home_team'] = np.select(conditions, choices_team)
    player_df['away_team'] = np.select(conditions, choices_opponent)

    # create merge_key which is home_team + away_team + year
    player_df['matchup_merge_key'] = player_df.apply(lambda row: '_'.join(sorted([row['home_team'], row['away_team']])), axis=1)

    # create season_merge_key which is sorted home_team + away_team + season
    player_df['season_merge_key'] = player_df.apply(lambda row: '_'.join(sorted([row['home_team'], row['away_team']] + [row['season']])), axis=1)

    # player_df['season_merge_key'] = player_df['matchup_merge_key'] + player_df['season']

    player_df['season_gameweek'] = player_df['season'] + '_' + player_df['gameweek'].astype(str)
    cols_order = ['player', 'team', 'season_gameweek', 'minutes', 'position'] + [col for col in player_df.columns if col not in ['player', 'team', 'season_gameweek', 'minutes', 'position']]
    player_df = player_df[cols_order]

    return player_df

def clean_dataframes(df):
    """Description: This function cleans the dataframes by removing unnecessary columns, reordering columns, creating new columns and renaming columns.
    
        Arguments:
            df {dataframe} -- dataframe to be cleaned
        
        Returns:
            df {dataframe} -- cleaned dataframe

    """
    drop_cols = ['score', 'match_report', 'notes']
    df = df.drop(drop_cols, axis=1)

    # reorder columns gameweek, home_team, home_score, home_xg, away_team, away_score, away_xg, date, season
    df = df[['gameweek', 'home_team', 'home_score', 'home_xg', 'away_team', 'away_score', 'away_xg', 'date', 'season']]

    # create a column for the winning_team, if draw then draw
    df['winning_team'] = np.where(df['home_score'] > df['away_score'], df['home_team'], np.where(df['home_score'] < df['away_score'], df['away_team'], 'draw'))

    # create a column for the losing_team, if draw then draw
    df['losing_team'] = np.where(df['home_score'] < df['away_score'], df['home_team'], np.where(df['home_score'] > df['away_score'], df['away_team'], 'draw'))

    # create match_id column which is applymap of home_team and away_team and season using uuid
    df['match_id'] = df[['home_team', 'away_team', 'season']].applymap(str).apply(lambda x: ''.join(x), axis=1).apply(lambda x: uuid.uuid5(uuid.NAMESPACE_DNS, x))

    df['matchup_merge_key'] = df[['home_team', 'away_team']].applymap(str).apply(lambda x: ''.join(x), axis=1).apply(lambda x: uuid.uuid5(uuid.NAMESPACE_DNS, x))
    df['season_merge_key'] = df[['home_team', 'away_team', 'season']].applymap(str).apply(lambda x: ''.join(x), axis=1).apply(lambda x: uuid.uuid5(uuid.NAMESPACE_DNS, x))

    # create team and opponent columns from home_team and away_team
    df['team'] = df['home_team']
    df['opponent'] = df['away_team']

    # create match_teams column from team and opponent where we sort the team and opponent alphabetically and join with _
    df['match_teams'] = df.apply(lambda row: '_'.join(sorted([row['team'].strip(), row['opponent'].strip()])), axis=1)
    
    # create season_match_teams column from match_teams and season
    df['season_match_teams'] = df['match_teams'] + '_' + df['season'].astype(str)

    # df['match_teams'] = df['team'] + '_' + df['opponent']

    # fillna() 0

    df = df.fillna(0)

    return df

def get_top_players(team, player_df, stat, top=5):
    """
    group the player_df by match_teams

    Parameters:
    team (str): The team to get players from.
    player_df (DataFrame): The player data.
    stat (str): The statistic to rank players by.
    top (int): The number of top players to return.

    Returns:
    DataFrame1: A DataFrame with the top players of team1 and their stats against the specific opponent
    """
    # check if team column exists in player_df
    if 'team' not in player_df.columns:
        raise ValueError('team column not in player_df')
    else:
        # Filter dataframe based on team
        player_df_team = player_df[player_df['team'] == team]

        # Top players by matchup
        top_players_matchup = player_df_team.groupby(['player']).agg({stat: 'sum'}).sort_values(by=stat, ascending=False).head(top)

        # Top players by season
        top_players_season = player_df_team.groupby(['player']).agg({stat: 'sum'}).sort_values(by=stat, ascending=False).head(top)

    return top_players_matchup, top_players_season


def get_teams_stats(df, team1, team2):
    stats_team1 = {
        'total_games': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_goals_scored': 0,
        'total_goals_conceded': 0,
        'xG For': 0,
        'xG Against': 0,
        'Clean Sheets': 0
    }
    stats_team2 = {
        'total_games': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_goals_scored': 0,
        'total_goals_conceded': 0,
        'xG For': 0,
        'xG Against': 0,
        'Clean Sheets': 0
    }

    df_filtered = df[df['match_teams'].isin([f'{team1}_{team2}', f'{team2}_{team1}'])]

    for index, row in df_filtered.iterrows():
        if row['home_team'] == team1:
            stats_team1['total_games'] += 1
            stats_team1['total_goals_scored'] += row['home_score']
            stats_team1['total_goals_conceded'] += row['away_score']
            stats_team1['xG For'] += row['home_xg']
            stats_team1['xG Against'] += row['away_xg']
            stats_team1['Clean Sheets'] += 1 if row['away_score'] == 0 else 0
            if row['winning_team'] == team1:
                stats_team1['total_wins'] += 1
            elif row['losing_team'] == team1:
                stats_team1['total_losses'] += 1

        if row['away_team'] == team1:
            stats_team1['total_games'] += 1
            stats_team1['total_goals_scored'] += row['away_score']
            stats_team1['total_goals_conceded'] += row['home_score']
            stats_team1['xG For'] += row['away_xg']
            stats_team1['xG Against'] += row['home_xg']
            stats_team1['Clean Sheets'] += 1 if row['home_score'] == 0 else 0
            if row['winning_team'] == team1:
                stats_team1['total_wins'] += 1
            elif row['losing_team'] == team1:
                stats_team1['total_losses'] += 1

        if row['home_team'] == team2:
            stats_team2['total_games'] += 1
            stats_team2['total_goals_scored'] += row['home_score']
            stats_team2['total_goals_conceded'] += row['away_score']
            stats_team2['xG For'] += row['home_xg']
            stats_team2['xG Against'] += row['away_xg']
            stats_team2['Clean Sheets'] += 1 if row['away_score'] == 0 else 0
            if row['winning_team'] == team2:
                stats_team2['total_wins'] += 1
            elif row['losing_team'] == team2:
                stats_team2['total_losses'] += 1

        if row['away_team'] == team2:
            stats_team2['total_games'] += 1
            stats_team2['total_goals_scored'] += row['away_score']
            stats_team2['total_goals_conceded'] += row['home_score']
            stats_team2['xG For'] += row['away_xg']
            stats_team2['xG Against'] += row['home_xg']
            stats_team2['Clean Sheets'] += 1 if row['home_score'] == 0 else 0
            if row['winning_team'] == team2:
                stats_team2['total_wins'] += 1
            elif row['losing_team'] == team2:
                stats_team2['total_losses'] += 1

    return stats_team1, stats_team2

def show_head2head_analysis(df_all_seasons, player_df):
    """Show head to head analysis between two teams for all seasons, a particular set of seasons, or a single seasons. The default is the most recent season. We get the user to select the season(s) they want to compare and then the teams they want to compare. We create a filtered dataframe based on the season(s) and teams selected. More specifically, from team and opponent selected by the user, we create a 'selected_teams' object that has the teams the user selected in alphabetical order. We filter the dataframe using the 'match_teams' column based on the 'selected_teams' object and then we group by player. Now we have a dataframe that contains all the players that played in the matches between the two teams selected by the user. We then create a multiselect for the user to select the stat(s) they want to compare. Default stats should be 'npxg', 'sca', 'gca'. 

    Args:
        df_all_seasons (dataframe): contains general match stats
        player_df (dataframe): contains player specific stats

    Returns:
        teams_dataframe: dataframe containing the stats of the players that played in the matches between the two teams selected by the user
        players_dataframe: dataframe containing the stats of the players that played in the matches between the two teams selected by the user
    """

    st.info('Select season(s) you want to compare. \nNote: Default is the most recent season')

    # create a multiselect for seasons to select one or many seasons but it should default to the most recent season
    season_selection = st.multiselect('Select Season(s)', reversed(sorted(df_all_seasons['season'].unique().tolist())), default=sorted(df_all_seasons['season'].unique().tolist())[-1])

    # Filter the DataFrame based on the season(s) selected
    df_filtered = df_all_seasons[df_all_seasons['season'].isin(season_selection)]

    st.info('Select the teams you want to compare')

    # Create a list of teams
    team_list = sorted(df_all_seasons['home_team'].unique().tolist())

    if 'team_selection1' not in st.session_state:
        st.session_state['team_selection1'] = team_list[0]

    if 'team_selection2' not in st.session_state:
        st.session_state['team_selection2'] = [team for team in team_list if team != st.session_state['team_selection1']][0]

    team_selection1 = st.selectbox('Select Primary Team', team_list, key='team_selection1')
    team_selection2 = st.selectbox('Select Opponent', [team for team in team_list if team != st.session_state['team_selection1']], key='team_selection2')

    # Apply the statistics function to the filtered DataFrame
    team1_stats, team2_stats = get_teams_stats(df_filtered, team_selection1, team_selection2)

    # Convert the dictionaries to a DataFrame for visualization
    df_stats = pd.DataFrame({team_selection1: team1_stats, team_selection2: team2_stats})

    # Display the DataFrame
    st.dataframe(df_stats)

    stat_list = [col for col in player_df.columns if (player_df[col].dtype == 'float64' or player_df[col].dtype == 'int64')]

    default_stats = ['npxg', 'sca', 'gca']
    default_stats = [stat for stat in default_stats if stat in stat_list]

    # Let the user select a stat
    selected_stats = st.multiselect('Select player stat', stat_list, default=default_stats)

    # If no stat is selected, use the default stats
    if not selected_stats:
        selected_stats = default_stats
        
    st.dataframe(player_df)

    # Update the get_top_players function with selected team and stats
    for stat in selected_stats:
        top_players_matchup, top_players_season = get_top_players(team_selection1, player_df, stat)
        st.write(f'Top players for {team_selection1} in matchup based on {stat}')
        st.dataframe(top_players_matchup)
        st.write(f'Top players for {team_selection1} in season based on {stat}')
        st.dataframe(top_players_season)

def main():
    # Load and process player data
    player_df = load_player_data()
    player_df = process_player_data(player_df)

    # Load and clean match data
    df_1992_2016 = pd.read_csv('data/historical_matches_reports-1992-2016.csv')
    df_2022_2023 = pd.read_csv('data/df_2023-07-25_12-17-41_2022-2023.csv')
    df_2021_2022 = pd.read_csv('data/df_2023-07-25_11-44-56_2021-2022.csv')
    df_2020_2021 = pd.read_csv('data/df_2023-07-25_11-08-29_2020-2021.csv')
    df_2019_2020 = pd.read_csv('data/df_2023-07-25_10-32-44_2019-2020.csv')
    df_2018_2019 = pd.read_csv('data/df_2023-07-25_09-57-12_2018-2019.csv')
    df_2017_2018 = pd.read_csv('data/df_2023-07-25_09-57-10_2017-2018.csv')

    df_all_seasons = pd.concat([df_2022_2023, df_2021_2022, df_2020_2021, df_2019_2020, df_2018_2019, df_2017_2018, df_1992_2016])
    df_all_seasons = clean_dataframes(df_all_seasons)

    # Show head to head analysis
    show_head2head_analysis(df_all_seasons, player_df)

if __name__ == "__main__":
    main()


# def show_head2head_analysis(df_all_seasons, player_df):
#     """Show head to head analysis between two teams for all seasons, a particular set of seasons, or a single seasons. The default is the most recent season. We get the user to select the season(s) they want to compare and then the teams they want to compare. We create a filtered dataframe based on the season(s) and teams selected. More specifically, from team and opponent selected by the user, we create a 'selected_teams' object that has the teams the user selected in alphabetical order. We filter the dataframe using the 'match_teams' column based on the 'selected_teams' object and then we group by player. Now we have a dataframe that contains all the players that played in the matches between the two teams selected by the user. We then create a multiselect for the user to select the stat(s) they want to compare. Default stats should be 'npxg', 'sca', 'gca'. 

#     Args:
#         df_all_seasons (dataframe): contains general match stats
#         player_df (dataframe): contains player specific stats

#     Returns:
#         teams_dataframe: dataframe containing the stats of the players that played in the matches between the two teams selected by the user
#         players_dataframe: dataframe containing the stats of the players that played in the matches between the two teams selected by the user
#     """
#     # Show head to head analysis between two teams for all seasons, a particular set of seasons, or a single seasons. The default is the most recent season. We get the user to select the season(s) they want to compare and then the teams they want to compare. We create a filtered dataframe based on the season(s) and teams selected. More specifically, from team and opponent selected by the user, we create a 'selected_teams' object that has the teams the user selected in alphabetical order. We filter the dataframe using the 'match_teams' column based on the 'selected_teams' object and then we group by player. Now we have a dataframe that contains all the players that played in the matches between the two teams selected by the user. We then create a multiselect for the user to select the stat(s) they want to compare. Default stats should be 'npxg', 'sca', 'gca'. 
