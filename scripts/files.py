# doc paths
import re
import os

# list comprehension that gets the file in 'data/projections/' where filename matches 'GW{is_number} Projections.csv' 
projections = [f'data/projections/{file}' for file in os.listdir('data/projections/') if re.match(r'GW\d+ Projections.csv', file)][0]

ros_ranks = [f'data/ros_ranks/{file}' for file in os.listdir('data/ros_ranks/') if re.match(r'Weekly ROS Ranks_GW\d+.csv', file)][0]

big5_players_csv = 'data/data_out/scraped_big5_data/player_table_big5.csv'
players_matches_csv = 'data/data_out/final_data/csv_files/results.csv'
teams_matches_csv = 'data/data_out/final_data/csv_files/players.csv'
big5_this_year = 'data/data_out/scraped_big5_data/big5_players_data_gw1.csv'

big5_squads_data = 'data/data_out/scraped_big5_data/squads_data/squad_table_2023-2024.csv'

pl_data_gw1 = 'data/data_out/scraped_big5_data/pl_data/pl_player_table_upto_gw1.csv'
pl_data_gw2 = 'data/data_out/scraped_big5_data/pl_data/pl_player_table_gw2_only.csv'

all_gws_data = 'data/data_out/scraped_big5_data/pl_data/pl_processed_table_through_all_gws.csv'

fdr_csv = 'data/data_out/final_data/csv_files/fdr.csv'

temp_gw1_fantrax_default = 'data/specific-csvs/Fantrax-Players-DraftPL Super-League.csv'

# match reports 
# data/data_out/scraped_big5_data/pl_data/all_shots_all_20230820.csv data/data_out/scraped_big5_data/pl_data/full_season_matchreports_20230820.csv

shots_data = 'data/data_out/scraped_big5_data/pl_data/all_shots_all_20230924.csv'
matches_data = 'data/data_out/scraped_big5_data/pl_data/full_season_matchreports_20230924.csv'

# files paths
data_out = 'data/data_out'

# data paths
scraped_big5_data = 'data/data_out/scraped_big5_data'

fx_gif = 'media/download_fantrax_data_demo.gif'

pl_2018_2023 = 'data/data_out/scraped_big5_data/pl_data/player_table_2018_to_2023.csv'