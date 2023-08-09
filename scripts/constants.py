import seaborn as sns

# constants.py
fbref_cats = ['stats', 'shooting', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'playingtime', 'misc']

fbref_leagues = ['Big5', 'ENG', 'ESP', 'ITA', 'GER', 'FRA']

seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

# colors 

color1 = "#2b2d42"
color2 = "#8d99ae"
color3 = "#edf2f4"
color4 = "#ef233c"
color5 = "#d90429"
color_dark1 = "#FFD6FF"

# gradients
cm = sns.diverging_palette(333, 33, s=99, l=57, as_cmap=True)

# doc paths
big5_players_csv = '/Users/hogan/dev/streamlit_proj_new/data/specific-csvs/player_table_big5.csv'

# urls
fbref_base_url = 'https://fbref.com/en/comps/'

fbref_current_year_url = 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'