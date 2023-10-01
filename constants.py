import seaborn as sns

# constants.py
colors = ['#140b04', '#1c1625', '#B82A2A']

divergent_colors = ['#100993', '#140b04', '#1c1625', '#B82A2A']

fbref_cats = ['stats', 'shooting', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'playingtime', 'misc']

fbref_leagues = ['Big5', 'ENG', 'ESP', 'ITA', 'GER', 'FRA']

seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

stats_cols = ['goals', 'assists', 'goals_assists', 'goals_pens', 'pens_made', 'pens_att', 'cards_yellow', 'cards_red', 'xg', 'npxg', 'xg_assist', 'npxg_xg_assist', 'progressive_carries', 'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90', 'goals_assists_per90', 'goals_pens_per90', 'goals_assists_pens_per90', 'xg_per90', 'xg_assist_per90', 'xg_xg_assist_per90', 'npxg_per90', 'npxg_xg_assist_per90']

shooting_cols = ['shots', 'shots_on_target', 'shots_on_target_pct', 'shots_per90', 'shots_on_target_per90', 'goals_per_shot', 'goals_per_shot_on_target', 'average_shot_distance', 'shots_free_kicks', 'pens_made', 'pens_att', 'xg', 'npxg', 'npxg_per_shot', 'xg_net', 'npxg_net']

passing_cols = ['passes_completed', 'passes', 'passes_pct', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'assists', 'xg_assist', 'pass_xa', 'xg_assist_net', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes']

passing_types_cols = ['passes', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_completed', 'passes_offsides', 'passes_blocked']

gca_cols = ['sca', 'sca_per90', 'sca_passes_live', 'sca_passes_dead', 'sca_take_ons', 'sca_shots', 'sca_fouled', 'sca_defense', 'gca', 'gca_per90', 'gca_passes_live', 'gca_passes_dead', 'gca_take_ons', 'gca_shots', 'gca_fouled', 'gca_defense']

defense_cols = ['tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocks', 'blocked_shots', 'blocked_passes', 'interceptions', 'tackles_interceptions', 'clearances', 'errors']

possession_cols = ['touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons', 'take_ons_won', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct', 'carries', 'carries_distance', 'carries_progressive_distance', 'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received']

playing_time_cols = ['minutes_per_game', 'minutes_pct', 'minutes_90s', 'games_starts', 'minutes_per_start', 'games_complete', 'games_subs', 'minutes_per_sub', 'unused_subs', 'points_per_game', 'on_goals_for', 'on_goals_against', 'plus_minus', 'plus_minus_per90', 'plus_minus_wowy', 'on_xg_for', 'on_xg_against', 'xg_plus_minus', 'xg_plus_minus_per90', 'xg_plus_minus_wowy']

misc_cols = ['cards_yellow', 'cards_red', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'crosses', 'interceptions', 'tackles_won', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

matches_drop_cols = ['shirtnumber', 'nationality', 'age', 'position_1', 'position_2', 'position_3']

matches_default_cols = ['player', 'GW', 'started', 'team', 'Pos']

other_default_cols = ['goals', 'assists', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'pass_xa', 'assisted_shots']
                        
matches_standard_cols = ['goals', 'assists', 'pens_made', 'pens_att', 'shots', 'shots_on_target', 'cards_yellow', 'cards_red', 'touches', 'tackles', 'interceptions', 'blocks', 'xg', 'npxg', 'xg_assist', 'sca', 'gca', 'passes_completed', 'passes', 'passes_pct', 'progressive_passes', 'carries', 'progressive_carries', 'take_ons', 'take_ons_won']

matches_passing_cols = ['passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area']

matches_pass_types = ['passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked']

matches_defense_cols = ['tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost', 'blocked_shots', 'blocked_passes', 'tackles_interceptions', 'clearances', 'errors', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'take_ons_won_pct', 'take_ons_tackled', 'take_ons_tackled_pct']

matches_possession_cols = ['carries_distance', 'carries_progressive_distance', 'carries_into_final_third', 'carries_into_penalty_area', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received']

matches_misc_cols = ['cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

matches_default_cols_rename = {'player': 'Player', 'GW': 'GW', 'started': 'Started', 'team': 'Team', 'Pos': 'Position'}

matches_standard_cols_rename = {'goals': 'Goals', 'assists': 'Assists', 'pens_made': 'Pens Made', 'pens_att': 'Pens Attempted', 'shots': 'Shots', 'shots_on_target': 'Shots on Target', 'cards_yellow': 'Yellow Cards', 'cards_red': 'Red Cards', 'touches': 'Touches', 'tackles': 'Tackles', 'interceptions': 'Int', 'blocks': 'Blocks', 'xg': 'xG', 'npxg': 'npxG', 'xg_assist': 'xA', 'sca': 'SCA', 'gca': 'GCA', 'passes_completed': 'Passes Completed', 'passes': 'Passes', 'passes_pct': 'Passes %', 'progressive_passes': 'Prog Passes', 'carries': 'Carries', 'progressive_carries': 'Prog Carries', 'take_ons': 'Take Ons', 'take_ons_won': 'Take Ons Won'}

matches_passing_cols_rename = {'passes_total_distance': 'Pass Dist', 'passes_progressive_distance': 'Prog Pass Dist', 'passes_completed_short': 'Passes Cmp Short', 'passes_short': 'Passes Short', 'passes_pct_short': 'Passes % Short', 'passes_completed_medium': 'Passes Cmp Medium', 'passes_medium': 'Passes Medium', 'passes_pct_medium': 'Passes % Medium', 'passes_completed_long': 'Passes Cmp Long', 'passes_long': 'Passes Long', 'passes_pct_long': 'Passes % Long', 'pass_xa': 'Pass xA', 'assisted_shots': 'Key Passes', 'passes_into_final_third': 'Passes into Final Third', 'passes_into_penalty_area': 'Passes into Penalty Area', 'crosses_into_penalty_area': 'Crosses into Penalty Area'}

matches_pass_types_rename = {'passes_live': 'Passes Live', 'passes_dead': 'Passes Dead', 'passes_free_kicks': 'Freekicks', 'through_balls': 'Through Balls', 'passes_switches': 'Passes Switches', 'crosses': 'Crosses', 'throw_ins': 'Throw Ins', 'corner_kicks': 'CKs', 'corner_kicks_in': 'CKs (inswinging)', 'corner_kicks_out': 'CKs (outswinging)', 'corner_kicks_straight': 'CKs (straight)', 'passes_offsides': 'Passes Offsides', 'passes_blocked': 'Passes Blocked'}

matches_defense_cols_rename = {'tackles_won': 'Tackles Won', 'tackles_def_3rd': 'Tackles Def 3rd', 'tackles_mid_3rd': 'Tackles Mid 3rd', 'tackles_att_3rd': 'Tackles Att 3rd', 'challenge_tackles': 'Challenge Tackles', 'challenges': 'Challenges', 'challenge_tackles_pct': 'Challenge Tackles %', 'challenges_lost': 'Challenges Lost', 'blocked_shots': 'Blocked Shots', 'blocked_passes': 'Blocked Passes', 'tackles_interceptions': 'Tackles + Int', 'clearances': 'Clearances', 'errors': 'Errors', 'touches_def_pen_area': 'Touches Def Pen Area', 'touches_def_3rd': 'Touches Def 3rd', 'touches_mid_3rd': 'Touches Mid 3rd', 'touches_att_3rd': 'Touches Att 3rd', 'touches_att_pen_area': 'Touches Att Pen Area', 'touches_live_ball': 'Touches Live Ball', 'take_ons_won_pct': 'Take Ons Won %', 'take_ons_tackled': 'Take Ons Tackled', 'take_ons_tackled_pct': 'Take Ons Tackled %'}

matches_possession_cols_rename = {'carries_distance': 'Carries Dist', 'carries_progressive_distance': 'Prog Carries Dist', 'carries_into_final_third': 'Carries into Final Third', 'carries_into_penalty_area': 'Carries into Penalty Area', 'miscontrols': 'Miscontrols', 'dispossessed': 'Dispossessed', 'passes_received': 'Passes Received', 'progressive_passes_received': 'Prog Passes Received'}

matches_misc_cols_rename = {'cards_yellow_red': 'Double Yellows', 'fouls': 'Fouls', 'fouled': 'Fouled', 'offsides': 'Offsides', 'pens_won': 'Pens Won', 'pens_conceded': 'Pens Conceded', 'own_goals': 'Own Goals', 'ball_recoveries': 'Ball Recoveries', 'aerials_won': 'Aerials Won', 'aerials_lost': 'Aerials Lost', 'aerials_won_pct': 'Aerials Won %'}

matches_rename_dict = {
    'player': 'Player',
    'GW': 'GW',
    'started': 'Started',
    'Position': 'Position',
    'team': 'Team',
    'goals': 'Goals',
    'assists': 'Assists',
    'pens_made': 'Pens Made',
    'pens_att': 'Pens Attempted',
    'shots': 'Shots',
    'shots_on_target': 'Shots on Target',
    'cards_yellow': 'Yellow Cards',
    'cards_red': 'Red Cards',
    'touches': 'Touches',
    'tackles': 'Tackles',
    'interceptions': 'Int',
    'blocks': 'Blocks',
    'xg': 'xG',
    'npxg': 'npxG',
    'xg_assist': 'xA',
    'sca': 'SCA',
    'gca': 'GCA',
    'passes_completed': 'Passes Completed',
    'passes': 'Passes',
    'passes_pct': 'Passes %',
    'progressive_passes': 'Prog Passes',
    'carries': 'Carries',
    'progressive_carries': 'Prog Carries',
    'take_ons': 'Take Ons',
    'take_ons_won': 'Take Ons Won',
    'passes_total_distance': 'Pass Dist',
    'passes_progressive_distance': 'Prog Pass Dist',
    'passes_completed_short': 'Passes Cmp Short',
    'passes_short': 'Passes Short',
    'passes_pct_short': 'Passes % Short',
    'passes_completed_medium': 'Passes Cmp Medium',
    'passes_medium': 'Passes Medium',
    'passes_pct_medium': 'Passes % Medium',
    'passes_completed_long': 'Passes Cmp Long',
    'passes_long': 'Passes Long',
    'passes_pct_long': 'Passes % Long',
    'pass_xa': 'Pass xA',
    'assisted_shots': 'Key Passes',
    'passes_into_final_third': 'Passes into Final Third',
    'passes_into_penalty_area': 'Passes into Penalty Area',
    'crosses_into_penalty_area': 'Crosses into Penalty Area',
    'passes_live': 'Passes Live',
    'passes_dead': 'Passes Dead',
    'passes_free_kicks': 'Freekicks',
    'through_balls': 'Through Balls',
    'passes_switches': 'Passes Switches',
    'crosses': 'Crosses',
    'throw_ins': 'Throw Ins',
    'corner_kicks': 'CKs',
    'corner_kicks_in': 'CKs (inswinging)',
    'corner_kicks_out': 'CKs (outswinging)',
    'corner_kicks_straight': 'CKs (straight)',
    'passes_offsides': 'Passes Offsides',
    'passes_blocked': 'Passes Blocked',
    'tackles_won': 'Tackles Won',
    'tackles_def_3rd': 'Tackles Def 3rd',
    'tackles_mid_3rd': 'Tackles Mid 3rd',
    'tackles_att_3rd': 'Tackles Att 3rd',
    'challenge_tackles': 'Challenge Tackles',
    'challenges': 'Challenges',
    'challenge_tackles_pct': 'Challenge Tackles %',
    'challenges_lost': 'Challenges Lost',
    'blocked_shots': 'Blocked Shots',
    'blocked_passes': 'Blocked Passes',
    'tackles_interceptions': 'Tackles + Int',
    'clearances': 'Clearances',
    'errors': 'Errors',
    'touches_def_pen_area': 'Touches Def Pen Area',
    'touches_def_3rd': 'Touches Def 3rd',
    'touches_mid_3rd': 'Touches Mid 3rd',
    'touches_att_3rd': 'Touches Att 3rd',
    'touches_att_pen_area': 'Touches Att Pen Area',
    'touches_live_ball': 'Touches Live Ball',
    'take_ons_won_pct': 'Take Ons Won %',
    'take_ons_tackled': 'Take Ons Tackled',
    'take_ons_tackled_pct': 'Take Ons Tackled %',
    'carries_distance': 'Carries Dist',
    'carries_progressive_distance': 'Prog Carries Dist',
    'carries_into_final_third': 'Carries into Final Third',
    'carries_into_penalty_area': 'Carries into Penalty Area',
    'miscontrols': 'Miscontrols',
    'dispossessed': 'Dispossessed',
    'passes_received': 'Passes Received',
    'progressive_passes_received': 'Prog Passes Received',
    'cards_yellow_red': 'Double Yellows',
    'fouls': 'Fouls',
    'fouled': 'Fouled',
    'offsides': 'Offsides',
    'pens_won': 'Pens Won',
    'pens_conceded': 'Pens Conceded',
    'own_goals': 'Own Goals',
    'ball_recoveries': 'Ball Recoveries',
    'aerials_won': 'Aerials Won',
    'aerials_lost': 'Aerials Lost',
    'aerials_won_pct': 'Aerials Won %'
    }

col_groups = {
    "Standard": stats_cols,
    "Shooting": shooting_cols,
    "Passing": passing_cols,
    "Defense": defense_cols,
    "Possession": possession_cols,
    "Miscellaneous": misc_cols,
    "Passing Types": passing_types_cols,
    "GCA": gca_cols,
    "Playing Time": playing_time_cols,
}

matches_col_groups = {
    "Standard": matches_standard_cols,
    "Passing": matches_passing_cols,
    "Defense": matches_defense_cols,
    "Possession": matches_possession_cols,
    "Miscellaneous": matches_misc_cols,
    "Passing Types": matches_pass_types
}

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

# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

