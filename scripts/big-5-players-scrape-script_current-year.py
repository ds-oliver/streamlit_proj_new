# %% [markdown]
# # Scraping **_Players data_** Big5 Euro Leagues from FBRef.com

# %%
import requests
import pandas as pd
from IPython.display import display, Markdown, clear_output
import time
import sys
from datetime import datetime
from tqdm import tqdm
from colorama import Fore, Style
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode


# %%
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# %%
categories_list = ['stats', 'shooting', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'playingtime', 'misc']
# categories_list = ['stats', 'shooting', 'passing']
single_season = [f"{season}-{str(season+1)}" for season == 2023]  # Include 2020
seasons_range = f"{seasons_list[0]}_{seasons_list[-1]}"

# %%
def normalize_encoding(df, column_name):
    """
    Normalizes the encoding of a pandas DataFrame column using the unidecode library.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the column to normalize.
        column_name (str): The name of the column to normalize.

    Returns:
        pandas DataFrame: A copy of the original DataFrame with the specified column normalized.
    """
    # Create a copy of the original DataFrame
    normalized_df = df.copy()

    # Normalize the encoding of the specified column using unidecode
    normalized_df[column_name] = normalized_df[column_name].apply(unidecode)

    return normalized_df



# %%
def filter_data_by_season_og(df):
    # Check if '90s' column exists in the DataFrame
    if '90s' not in df.columns:
        raise ValueError("'90s' column not found in the DataFrame.")

    warnings.filterwarnings(
        'ignore', message='In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array.')

    warnings.filterwarnings(
        'ignore', message='FutureWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or , if columns are non-unique')

    # Create an empty dataframe to store the filtered data
    filtered_df = pd.DataFrame()

    season_data_filtered = pd.DataFrame()  # Define the variable before the loop

    # Loop through each league
    for league in df['League'].unique():
        # Skip the Champions League
        if league == 'Champions League':
            continue

        # Loop through each season for that league
        for season in df.loc[df['League'] == league, 'Season'].unique():
            # Get the data for that specific season and league
            season_data = df.loc[(df['League'] == league)
                                 & (df['Season'] == season)].copy()
            season_data.loc[:, '90s'] = pd.to_numeric(
                season_data.loc[:, '90s'], errors='coerce')

            # Check if '90s' column exists in the season data
            if '90s' not in season_data.columns:
                print(f"'90s' column not found for {league} {season}. Skipping.")
                continue

            # Calculate the q1 for the 90s column for that specific season and league
            q2 = season_data['90s'].quantile(0.50)
            # print(q2)

            # Filter the data to only include rows where the 90s column value is greater than the q1 value for that specific season and league
            season_data_filtered = season_data.loc[season_data['90s'] > q2].copy()

            # Append the filtered data to the overall filtered dataframe
            filtered_df = pd.concat([filtered_df, season_data_filtered])

            # print(f"Filtered data for {league}, {season}")
            

    timestamp = time.strftime("%Y-%m-%d_%H-%M")

    # Create the output file name using the formatted date
    output_file_name = f"/Users/hogan/dev/fbref/output/filtered_output_{timestamp}.csv"
    # Save the DataFrame to a CSV file with the output file name
    filtered_df.to_csv(output_file_name, index=False)
    print("DataFrame saved to:", output_file_name)
    print("Function completed:", timestamp)

    # View the resulting filtered data
    return filtered_df, output_file_name

# %%
def filter_data_by_season(df):
    # Handle warnings
    warnings.filterwarnings('ignore', message='In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array.')

    # Convert '90s' to numeric, coercing errors
    df.loc[:, '90s'] = pd.to_numeric(df.loc[:, '90s'], errors='coerce')

    # Create an empty dataframe to store the filtered data
    filtered_df = pd.DataFrame()

    # Use groupby to iterate over each league and season
    grouped = df.groupby(['League', 'Season'])
    for name, group in grouped:
        league, season = name

        # Skip if '90s' column is missing
        if '90s' not in group.columns:
            print(f"'90s' column not found for {league} {season}. Skipping.")
            continue

        # Skip the Champions League
        if league == 'Champions League':
            continue

        # Calculate the q2 value
        q2 = group['90s'].quantile(0.50)

        # Filter the data
        season_data_filtered = group.loc[group['90s'] > q2].copy()

        # Append the filtered data to the overall filtered dataframe
        filtered_df = pd.concat([filtered_df, season_data_filtered])

    return filtered_df


# %%
with tqdm(total=len(seasons_list) * len(categories_list), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)) as pbar:

    all_season_data = []  # List to store dataframes for each season

    for season in seasons_list:
        clear_output(wait=True)
        print(f"Scraping season: {season}")
        player_table = None
        scraped_columns_base = []
        
        for cat_index, cat in enumerate(categories_list):
            # Handle most recent season differently
            if season == '2023-24':
                url = f'https://fbref.com/en/comps/Big5/{cat}/players/Big-5-European-Leagues-Stats'
            else:
                url = f'https://fbref.com/en/comps/Big5/{season}/{cat}/players/{season}-Big-5-European-Leagues-Stats'
                
            print(f"A. Scraping {cat} player data for {season} - {url}")
            resp = requests.get(url).text
            htmlStr = resp.replace('<!--', '')
            htmlStr = htmlStr.replace('-->', '')
            
            if cat == 'playingtime':
                temp_df = pd.read_html(htmlStr, header=1)[0]
            else:
                temp_df = pd.read_html(htmlStr, header=1)[1]
                
            temp_df = temp_df[temp_df['Rk'] != 'Rk']  # Remove duplicate headers
            temp_df['Season'] = season  # Add season column
            temp_df['Season'] = temp_df['Season'].str[-4:]

            if player_table is None:
                player_table = temp_df
                scraped_columns_base = player_table.columns.tolist()
            else:
                new_columns = [col for col in temp_df.columns if col not in scraped_columns_base and col not in ['Player', 'Squad', 'Season']]
                temp_df = temp_df[['Player', 'Squad', 'Season'] + new_columns]
                player_table = pd.merge(player_table, temp_df, on=['Player', 'Squad', 'Season'], how='left')
                scraped_columns_base += new_columns
            
            print(f"Finished scraping {cat} data for {season}, DataFrame shape: {temp_df.shape}")        
            print(f"After operations and/or merging, player_table shape: {player_table.shape}")
            
            pbar.set_description(f"Processing item {cat_index+1}. ETR: {pbar.format_dict['remaining_s'] if 'remaining_s' in pbar.format_dict else None}")
            pbar.update(1)

        all_season_data.append(player_table)

    # Concatenate all seasons data into one DataFrame
    final_player_table = pd.concat(all_season_data, ignore_index=True)
    
    # reorder columns so that 'Season' is 3rd column
    cols = final_player_table.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    final_player_table = final_player_table[cols]
    # check first 10 columns
    print(final_player_table.columns.tolist()[:10])

    print("\nFinal DataFrame shape:", final_player_table.shape)

# Get the range of seasons
seasons_range = f"{seasons_list[0][-4:]}_to_{seasons_list[-1][-4:]}"

# Get current date and time
now = datetime.now()

# Format the current date and time to be used in the filename
timestamp = now.strftime("%m-%d_%H-%M-%S")

# Combine parts to create the filename
filename = f"/Users/hogan/dev/fbref/scripts/rfx_scrape/output/player_table_{seasons_range}_{timestamp}.csv"

# print cols of final_player_table
print(final_player_table.columns.tolist())

# Save the DataFrame to a CSV file
final_player_table.to_csv(filename, index=False)

# print filename with fstring
print(f"CSV output available here: {filename}")

# %%
# Get the range of seasons
seasons_range = f"{seasons_list[0][-4:]}_to_{seasons_list[-1][-4:]}"

# Get current date and time
now = datetime.now()

# Format the current date and time to be used in the filename
timestamp = now.strftime("%m-%d_%H-%M-%S")

# Combine parts to create the filename
filename = f"/Users/hogan/dev/fbref/scripts/rfx_scrape/output/player_table_{seasons_range}_{timestamp}.csv"

# Save the DataFrame to a CSV file
final_player_table.to_csv(filename, index=False)

# print filename with fstring
print(f"CSV output available here: {filename}")

# %%
df = final_player_table.copy()

# %%
# # read csv just created
# filename = '/Users/hogan/dev/fbref/scripts/rfx_scrape/output/player_table_2014_to_2023_06-26_18-54-44.csv'
# df = pd.read_csv(filename)
# print(df.shape)

# %%
# seasons_range = f"{seasons_list[0][-4:]}_to_{seasons_list[-1][-4:]}"

# now = datetime.now()

# # Format the current date and time to be used in the filename
# timestamp = now.strftime("%m-%d_%H-%M-%S")

# # Combine parts to create the filename
# filename = f"/Users/hogan/dev/fbref/scripts/rfx_scrape/output/player_table_{seasons_range}_{timestamp}.csv"

# # Save the DataFrame to a CSV file
# df_new_order.to_csv(filename, index=False)

# # print filename with fstring
# print(f"CSV output available here: {filename}")

# %%
def categorize_positions(df):
    # drop df[df['Pos'] == 0]
    df = df[df['Pos'] != 0]
    # drop gk rows
    df = df[df['Pos'] != 'GK']
    # create a dictionary of positions and their categories out of items in this list: ['FW' 'DF,MF' 'GK' 'DF' 'MF' 'FW,MF' 'DF,FW' 'DF,GK' 'FW,GK' 'MF,DF' 'MF,FW' 'FW,DF']
    positions_dict = {'FW': 'Forward', 'DF,MF': 'Defender', 'DF': 'Defender', 'MF': 'Midfielder', 'FW,MF': 'Forward', 'DF,FW': 'Defender', 'DF,GK': 'Defender', 'FW,GK': 'Forward', 'MF,DF': 'Midfielder', 'MF,FW': 'Midfielder', 'FW,DF': 'Forward'}
    # create a new column called 'Position Category' and fill it with the values from the dictionary
    df['Position Category'] = df['Pos'].map(positions_dict)
    
    print(df.columns.tolist())
    # return the dataframe
    return df

# %%
def clean_nation(nation):
    if isinstance(nation, str):
        return nation[-3:]
    else:
        return nation
    
def clean_comp(comp):
    if isinstance(comp, str):
        # strip the first 3 characters
        return comp[3:]
    else:
        return comp
    

# %%
column_name_map = {
    'xG.1': 'xG_per90',
    'Starts': 'games_started',
    'xAG.1': 'xAG_per90',
    'xG+xAG': 'expected_goals_plus_assisted_goals_per90',
    'npxG.1': 'non_penalty_expected_goals_per90',
    'npxG+xAG.1': 'non_penalty_expected_goals_plus_assisted_goals_per90',
    '# Pl': 'number_of_players_used',
    'npxG/Sh': 'non_penalty_expected_goals_per_shot',
    'G-xG': 'goals_minus_expected_goals',
    'np:G-xG': 'non_penalty_goals_minus_expected_goals',
    'Cmp%.1': 'short_pass_completion_percent',
    'Cmp%.2': 'medium_pass_completion_percent',
    'Cmp%.3': 'long_pass_completion_percent',
    'xA': 'expected_assists',
    'MP': 'matches_played',
    'Mn/MP': 'minutes_per_match',
    'Min%': 'minutes_played_percent',
    'Mn/Start': 'minutes_per_start',
    'Compl': 'complete_matches_played',
    'Subs': 'substitute_appearances',
    'Mn/Sub': 'minutes_per_substitute_appearance',
    'unSub': 'games_as_unused_substitute',
    'PPM': 'team_points_per_match_average',
    'onG': 'team_goals_scored_on_pitch',
    'onGA': 'team_goals_allowed_on_pitch',
    '+/-': 'goals_scored_minus_goals_allowed',
    '+/-90': 'goals_scored_minus_goals_allowed_per90',
    'onxG': 'team_expected_goals_scored_on_pitch',
    'onxGA': 'team_expected_goals_allowed_on_pitch',
    'On-Off': 'net_team_goals_on_pitch_minus_team_goals_allowed_off_pitch_per90',
    'xG+/-': 'team_expected_goals_on_pitch',
    'xG+/-90': 'team_expected_goals_minus_expected_goals_allowed_while_on_pitch_per90',
    'On-Off.1': 'net_team_expected_goals_on_pitch_minus_team_expected_goals_allowed_off_pitch_per90',
    'Min': 'minutes_played',
    'Ast': 'assists',
    'Gls': 'goals',
    'G+A': 'goals_and_assists',
    'G-PK': 'goals_minus_penalty_kicks',
    'PK': 'penalty_kicks_scored',
    'PKatt': 'penalty_kicks_attempted',
    'CrdY': 'yellow_cards',
    'CrdR': 'red_cards',
    'Gls.1': 'goals_per90',
    'Ast.1': 'assists_per90',
    'G+A.1': 'goals_and_assists_per90',
    'G-PK.1': 'goals_less_penalties_scored_per90',
    'G+A-PK': 'goals_and_assists_minus_penalty_kicks_scored',
    'Sh': 'shots',
    'SoT': 'shots_on_target',
    'SoT%': 'percent_of_shots_on_target',
    'Sh/90': 'shots_per_90_minutes',
    'SoT/90': 'shots_on_target_per_90_minutes',
    'G/Sh': 'goals_divided_by_shots',
    'G/SoT': 'goals_divided_by_shots_on_target',
    'Dist': 'passing_distance_covered',
    'Cmp': 'total_passes_completed',
    'Att': 'total_passes_attempted',
    'Cmp%': 'total_percent_pass_completion',
    'TotDist': 'passes_total_distance',
    'PrgDist': 'passes_distance_progressed',
    'Cmp.1': 'short_passes_completed',
    'Att.1': 'short_passes_attempted',
    'Cmp.2': 'medium_passes_completed',
    'Att.2': 'medium_passes_attempted',
    'Cmp.3': 'long_passes_completed',
    'Att.3': 'long_passes_attempted',
    'A-xAG': 'assists_plus_expected_assisted_goals',
    'KP': 'key_passes',
    '1/3': 'passes_into_key_areas',
    'PPA': 'passes_into_penalty_area',
    'CrsPA': 'crosses_into_penalty_area',
    'SCA': 'shot_creating_actions',
    'SCA90': 'shot_creating_actions_per_90_minutes',
    'PassLive': 'live_passes_leading_to_shots',
    'PassDead': 'dead_balls_leading_to_shots',
    'TO': 'take_ons_leading_to_shots',
    'Fld': 'fouls_drawn_leading_to_shots',
    'Def': 'defensive_action_leading_to_shots',
    'GCA': 'goal_creating_actions',
    'GCA90': 'goal_creating_actions_per_90_minutes',
    'PassLive.1': 'live_passes_leading_to_goals',
    'PassDead.1': 'dead_balls_leading_to_goals',
    'TO.1': 'take_ons_leading_to_goals',
    'Sh.1': 'shots_leading_to_goals',
    'Fld.1': 'fouls_drawn_leading_to_goals',
    'Def.1': 'defensive_actions_leading_to_goals',
    'Tkl': 'number_of_players_tackled',
    'TklW': 'tackles_won',
    'Def 3rd': 'tackles_in_defensive_third',
    'Mid 3rd': 'tackles_in_middle_third',
    'Att 3rd': 'tackles_in_attacking_third',
    'Tkl.1': 'number_of_dribblers_tackled',
    'Tkl%': 'percent_of_dribblers_successfully_tackled',
    'Lost': 'challenges_lost',
    'Blocks': 'balls_blocked',
    'Pass': 'passes_blocked',
    'Int': 'interceptions',
    'Tkl+Int': 'tackles_plus_interceptions',
    'Clr': 'clearances',
    'Err': 'errors_leading_to_opponent_shot',
    'Touches': 'touches',
    'Def Pen': 'touches_in_defensive_penalty_area',
    'Att Pen': 'touches_in_attacking_penalty_area',
    'Live': 'live-ball_touches',
    'Succ': 'successful_take_ons',
    'Succ%': 'percent_successful_take_ons',
    'Tkld': 'times_tackled_during_take_ons',
    'Tkld %': 'percent_times_tackled_during_take_ons',
    'Carries': 'times_ball_under_control',
    'CPA': 'carries_into_penalty_area',
    'Mis': 'miscontrols',
    'Dis': 'dispossessed',
    'Rec': 'passes_received',
    '2CrdY': 'second_yellow_card',
    'Fls': 'fouls_committed',
    'Off': 'offsides_offenses',
    'Crs': 'crosses',
    'PKwon': 'penalty_kicks_won',
    'PKcon': 'penalty_kicks_conceded',
    'OG': 'own_goals',
    'Dead': 'dead_ball_passes',
    'FK': 'freekick_passes',
    'TB': 'through_balls',
    'Sw': 'switches',
    'TI': 'throwins',
    'CK': 'corner_kicks',
    'In': 'inswinging_corner_kicks',
    'Out': 'outswinging_corner_kicks',
    'Str': 'straight_corner_kicks',
    'xG': 'expected_goals',
    'npxG': 'non-penalty_expected_goals',
    'xAG': 'expected_assisted_goals',
    'npxG+xAG': 'non-penalty_expected_goals_and_assisted_goals',
    'PrgC': 'progressive_passes_completed',
    'PrgP': 'progressive_carries',
    'PrgR': 'progressive_passes_received',
}

# Rename column names based off dict above
df.rename(columns=column_name_map, inplace=True)

print(df.columns.tolist())

# %%
cleaned_df = df.copy()

# %%
cleaned_df['Nation'] = cleaned_df['Nation'].apply(clean_nation)

# %%
cleaned_df['Comp'] = cleaned_df['Comp'].apply(clean_comp)

# %%
print(cleaned_df.columns.tolist())

# %%
cleaned_df['League'] = cleaned_df['Comp']
cleaned_df['League']

# %%
cleaned_df = normalize_encoding(cleaned_df, 'Player')
cleaned_df = normalize_encoding(cleaned_df, 'Squad')
cleaned_df = normalize_encoding(cleaned_df, 'League')

# strip 'League' column whitespace
cleaned_df['League'] = cleaned_df['League'].str.strip()

# %%
cleaned_df = categorize_positions(df)

# %%
# List of your first 10 columns in the new order
new_order = ['Player', 'Season', 'Squad', 'League', '90s', 'games_started', 'Position Category', 'Age', 'goals', 'assists']
other_columns = [col for col in cleaned_df.columns if col not in new_order]
# Create a new DataFrame with the first 10 columns reordered
df_new_order = cleaned_df[new_order]

df_reordered = cleaned_df[new_order + other_columns]

# %%
print(df_reordered.columns.tolist())

# %%
df_reordered['Position Category']

# %%
df_reordered.to_csv(f'/Users/hogan/dev/fbref/scripts/rfx_scrape/output/cleaned_players_table_{seasons_range}_{timestamp}.csv', index=False)

# %%
cleaned_df['League'].unique()

# %%
filtered_data, output_file_name = filter_data_by_season_og(cleaned_df)
print(filtered_data['90s'].unique().round(0))

# %%
filtered_df = filter_data_by_season(cleaned_df)

# %%
print(cleaned_df.shape)
print(filtered_df.shape)


# %%
def conduct_tests(df):
    print("\nConducting tests...\n")
    
    # Test for missing values
    print("Testing for missing values in '90s' column...")
    if df['90s'].isnull().any():
        print("Test result: There are missing values in the '90s' column.\n")
    else:
        print("Test result: There are no missing values in the '90s' column.\n")
        
    # Test for data type issues
    print("Testing for data type issues in '90s' column...")
    if df['90s'].dtype not in ['float64', 'int64']:
        print(f"Test result: The '90s' column is not a numeric type, it's {df['90s'].dtype}.\n")
    else:
        print("Test result: The '90s' column is of numeric type.\n")
    
    # Test for outliers or incorrect data
    print("Testing for outliers in '90s' column...")
    Q1 = df['90s'].quantile(0.25)
    Q3 = df['90s'].quantile(0.75)
    IQR = Q3 - Q1
    if ((df['90s'] < (Q1 - 1.5 * IQR)) | (df['90s'] > (Q3 + 1.5 * IQR))).any():
        print("Test result: There are potential outliers in the '90s' column.\n")
    else:
        print("Test result: There are no potential outliers in the '90s' column.\n")
        
    # Test for function execution
    print("Testing function execution...")
    try:
        filtered_df = filter_data_by_season(df)
        print("Test result: The function executed successfully.\n")
    except Exception as e:
        print(f"Test result: The function did not execute successfully. Error: {str(e)}\n")
        
    # Test for indexing issues
    print("Testing for indexing issues...")
    if isinstance(df.index, pd.MultiIndex):
        print("Test result: The DataFrame has a MultiIndex, which might be causing issues.\n")
    else:
        print("Test result: The DataFrame does not have a MultiIndex.\n")

# Run the tests
conduct_tests(cleaned_df)


# %%
filtered_df['90s'].unique().round(0)

# %%
def print_median_90s_by_season(df):
    # Convert '90s' to numeric, coercing errors
    df.loc[:, '90s'] = pd.to_numeric(df.loc[:, '90s'], errors='coerce')

    # Use groupby to iterate over each league and season
    grouped = df.groupby(['League', 'Season'])
    for name, group in grouped:
        league, season = name

        # Calculate the q2 value
        q2 = group['90s'].quantile(0.50)

        # Print the league, season and median '90s' value
        print(f"League: {league}, Season: {season}, Median '90s': {q2}")
        
print_median_90s_by_season(filtered_df)


# %%
# visualize histogram of 90s played for each dataframe on black chart background
pastel_palette = ['#FFC8B4', '#C8FFB4']
sns.set_palette(sns.color_palette(pastel_palette))
sns.set_style('dark')   
# plot histogram of 90s played for each dataframe using seaborn
fig, ax = plt.subplots(figsize=(12, 8))
# adjust color of lines and fill
ax = sns.histplot(data=cleaned_df, x='90s', ax=ax, kde=True, stat='density', label='Before Filter', edgecolor='black', linewidth=1.5, fill=True, alpha=0.5, color='#FFC8B4')
ax = sns.histplot(data=filtered_df, x='90s', ax=ax, kde=True, stat='density', label='After Filter', edgecolor='black', linewidth=1.5, fill=True, alpha=0.25, color='#C8FFB4')     
# set title
ax.set_title('Distribution of 90s Played', fontsize=20, fontweight='bold')
# set x-axis label
ax.set_xlabel('90s Played', fontsize=16, fontweight='bold')
# set y-axis label
ax.set_ylabel('Number of Players', fontsize=16, fontweight='bold')
# set tick font size
ax.tick_params(labelsize=12)
# remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# add legend
ax.legend(fontsize=14)

# %% [markdown]
# ## Import DF if restarting kernel after recent scrape

# %%
# List of your first 10 columns in the new order
new_order = ['Player', 'Season', 'Squad', 'League', '90s', 'Starts', 'Position Category', 'Age', 'Gls', 'Ast']

# Create a new DataFrame with the first 10 columns reordered
df_new_order = df[new_order]

# Append the remaining columns in their original order
df_new_order = pd.concat([df_new_order, df[df.columns[~df.columns.isin(new_order)]]], axis=1)

df = df_new_order.copy()

# %%
# perform data cleaning and processing on the dataframe
df = df.apply(pd.to_numeric, errors='ignore') # convert all columns to numeric
df = df.fillna(0)   # fill NaN values with 0
df = df.replace(np.inf, 0)  # replace infinite values with 0


# %%
filtered_df = filter_data_by_season(df)

# %%
print(df.head(10))


