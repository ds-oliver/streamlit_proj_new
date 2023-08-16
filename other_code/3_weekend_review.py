import requests
from requests.exceptions import RequestException, Timeout
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import os
import sys
import time
import datetime
import json
import logging
import logging.config
import yaml
import random
import itertools
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
from pathlib import Path
from unidecode import unidecode
from tqdm import tqdm
from IPython.display import clear_output


def get_url_table_ids():
    url_table_ids = {}
    # modify the test year values to suit your needs
    start_year = 2022
    end_year = 2023
    season = f"{start_year}-{end_year}"

    # print the years being scraped
    print(f"Scraping data for season(s) {start_year}-{end_year}.")

    for year in range(start_year, end_year):
        if start_year == 2023:
            current_year_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
            current_year_table_id = "sched_2023-2024_9_1"
            url_table_ids[current_year_url] = current_year_table_id
        else:            
            year_start_str = str(year)
            year_end_str = str(year + 1)
            url = f"https://fbref.com/en/comps/9/{year_start_str}-{year_end_str}/schedule/{year_start_str}-{year_end_str}-Premier-League-Scores-and-Fixtures"
            table_id = f"sched_{year_start_str}-{year_end_str}_9_1"
            url_table_ids[url] = table_id
            season = f"{year_start_str}-{year_end_str}"

    return url_table_ids, season

def parse_team_stats_table(table):
    column_headers = []
    data = []

    rows = table.find_all('tr')
    if rows:
        first_header_row_processed = False
        for i, row in enumerate(rows):
            # Ignore over_header rows and the last row which is the total
            if 'over_header' in row.get('class', []) or i == len(rows) - 1:
                continue

            headers = row.find_all('th')
            if headers and not first_header_row_processed:
                # Check if the th tags are direct children of the table
                if headers[0].parent.name == 'tr' and headers[0].parent.parent.name == 'thead':
                    # Extract column headers
                    column_headers = [header.get(
                        'data-stat') for header in headers]
                    # Check if column_headers are unique
                    if len(column_headers) != len(set(column_headers)):
                        raise ValueError(
                            "Duplicate column headers detected.")

                    first_header_row_processed = True
                continue  # Skip the rest of the loop for this iteration

            # Process data rows
            # Changed to find both 'th' and 'td' cells
            cells = row.find_all(['th', 'td'])
            if cells:
                output_row = {}

                for i, header in enumerate(column_headers):
                    if i < len(cells):
                        output_row[header] = cells[i].text
                    else:
                        # Set to None if cell does not exist
                        output_row[header] = None

                data.append(output_row)

    # Apply unidecode to player names
    for row in data:
        if 'player' in row:
            row['player'] = unidecode.unidecode(row['player'])

    print(f'Column Headers: {column_headers}')

    return column_headers, data
    
def parse_all_shots_table(table):
    shots_all_cols = ['minute', 'player', 'team', 'xg_shot', 'psxg_shot', 'outcome',
                        'distance', 'body_part', 'notes', 'sca_1_player', 'sca_1_type',
                        'sca_2_player', 'sca_2_type']

    data = []
    rows = table.find_all('tr')
    for row in rows:
        # Ignore separator row
        if 'spacer' in row.get('class', []):
            continue

        output_row = {}
        for header in shots_all_cols:
            cell = row.find('td', {'data-stat': header})
            if cell is None and header == 'minute':
                cell = row.find('th', {'data-stat': header})
            if cell:
                output_row[header] = cell.text

        # Check that the row contains at least 3 columns to be included and player names are not None
        if len(output_row) < 3 or None in [output_row.get('player'), output_row.get('sca_1_player')]:
            continue

        # Apply unidecode to player names
        output_row['player'] = unidecode.unidecode(output_row['player'])
        output_row['sca_1_player'] = unidecode.unidecode(
            output_row['sca_1_player'])
        # Check that sca_2_player is not None before unidecoding
        if output_row.get('sca_2_player'):
            output_row['sca_2_player'] = unidecode.unidecode(
                output_row['sca_2_player'])

        data.append(output_row)

    return shots_all_cols, data

def get_season_data(url, headers_request):
    response = requests.get(url, headers=headers_request, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def prepare_season_dataframe(data, headers, table_id, url):
    table = data.find("table", {"id": table_id})
    if table is None:
        return None, headers, None
    if not headers:
        table_head = table.find("thead")
        header_cols = table_head.find_all("th")
        headers = [col.get("data-stat") for col in header_cols]
        headers.append("season")

    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    season_data = []
    for row in rows:
        if row.has_attr("class") and "thead" in row["class"]:
            continue
        cols = [ele.text.strip() for ele in row.find_all(["th", "td"])]
        while len(cols) < len(headers):
            cols.insert(-1, np.nan)
        cols[-1] = table_id.split("_")[1].split("-")[0]
        season_data.append(cols)

    df = pd.DataFrame(season_data, columns=headers)
    df = df.dropna(subset=["score"])
    df = df[df["score"] != ""]
    df[["home_score", "away_score"]] = df["score"].str.split("–", expand=True)
    df["home_score"] = df["home_score"].replace("", np.nan).astype(float)
    df["away_score"] = df["away_score"].replace("", np.nan).astype(float)
    df = df.dropna(subset=["home_score"])
    # df = df.drop(["dayofweek", "start_time", "attendance",
    #              "venue", "referee"], axis=1)

    # print (df)
    print(f"Successfully scraped {url}")

    return df, headers, table_body

def cleaning_home_away_reports_tables(df):
    """
    Function to clean and reformat specific columns of a dataframe.
    
    Args:
        df (pandas.DataFrame): input dataframe with columns 'age', 'nationality', and 'position'.
        
    Returns:
        df_copy (pandas.DataFrame): cleaned dataframe.
    """
    df_copy = df.copy()

    # check if 'age' and 'nationality' columns exist and in expected format
    if 'age' in df_copy and df_copy['age'].str[:2].str.isnumeric().all():
        # clean age column so that it keeps only first 2 digits
        df_copy['age'] = df_copy['age'].str[:2]
    else:
        print("Age column is not in expected format or does not exist.")

    if 'nationality' in df_copy and df_copy['nationality'].str[-3:].str.isalpha().all():
        # clean nationality column so that it keeps only the last 3 characters
        df_copy['nationality'] = df_copy['nationality'].str[-3:]
    else:
        print("Nationality column is not in expected format or does not exist.")

    if 'position' in df_copy:
        # Split the 'position' column into as many new columns as needed
        positions = df_copy['position'].str.split(',', expand=True)
        for i in range(positions.shape[1]):
            df_copy[f'position_{i+1}'] = positions[i].str.strip()
        # Drop the original 'position' column
        df_copy.drop('position', axis=1, inplace=True)
    else:
        print("Position column does not exist.")

    # strip white spaces from the beginning and end of all columns values
    df_copy = df_copy.applymap(
        lambda x: x.strip() if isinstance(x, str) else x)

    return df_copy

def scrape_match_report(match_report_url, base_url, team_table_ids):
    """
    Function scrapes the match report page and returns a dataframe with the data.

    Args:
        match_report_url (str): url of the match report page
        base_url (str): base url of the website
        team_table_ids (list): list of table ids to scrape from the page

    Returns:
        pd.DataFrame: dataframe with the scraped data
    """
    
    # Construct full URL
    full_url = f"{base_url}/{match_report_url.lstrip('/')}"
    response = requests.get(full_url)

    soup = BeautifulSoup(response.text, 'html.parser')
    gameweek_search = re.search('\(Matchweek (\d{1,2})\)', response.text)
    gameweek = int(gameweek_search.group(1)) if gameweek_search else None

    merge_cols = ['player']
    encountered_columns_home = []
    encountered_columns_away = []
    home_team_df = None
    away_team_df = None

    try:
        # Process each team table id
        for table_id in team_table_ids:
            # Process home team table
            home_team_table = soup.find(
                'table', {'id': lambda x: x and x.startswith('stats_') and x.endswith(table_id)})
            home_team_name = home_team_table.find_previous(
                'h2').text.replace(' Player Stats', '')
            home_team_headers, home_team_stats = parse_team_stats_table(
                home_team_table)
            # Get unique headers and merge dataframes
            unique_headers_home = [
                h for h in home_team_headers if h == 'player' or h not in encountered_columns_home]
            encountered_columns_home += unique_headers_home
            temp_home_df = pd.DataFrame(
                home_team_stats, columns=unique_headers_home)
            home_team_df = temp_home_df if home_team_df is None else pd.merge(
                home_team_df, temp_home_df, on=merge_cols, how='outer')

            # Process away team table
            away_team_table = soup.find('table', {'id': lambda x: x and x.startswith(
                'stats_') and x.endswith(table_id) and x != home_team_table['id']})
            away_team_name = away_team_table.find_previous(
                'h2').text.replace(' Player Stats', '')
            away_team_headers, away_team_stats = parse_team_stats_table(
                away_team_table)
            # Get unique headers and merge dataframes
            unique_headers_away = [
                h for h in away_team_headers if h == 'player' or h not in encountered_columns_away]
            encountered_columns_away += unique_headers_away
            temp_away_df = pd.DataFrame(
                away_team_stats, columns=unique_headers_away)
            away_team_df = temp_away_df if away_team_df is None else pd.merge(
                away_team_df, temp_away_df, on=merge_cols, how='outer')

        # Process shots all table
        shots_all_table = soup.find('table', {'id': 'shots_all'})
        shots_all_headers, shots_all_data = parse_all_shots_table(
            shots_all_table)
        shots_all_df = pd.DataFrame(shots_all_data, columns=shots_all_headers)

        home_team_df['home'] = True
        away_team_df['home'] = False

        home_team_df['team'] = home_team_name
        away_team_df['team'] = away_team_name

        home_team_df['opponent'] = away_team_name
        away_team_df['opponent'] = home_team_name

        # Add gameweek column
        home_team_df['gameweek'] = gameweek
        away_team_df['gameweek'] = gameweek

        return shots_all_df, home_team_df, away_team_df
    except Exception as e:
        return None, None, None


def process_and_save_scraped_data(match_report_url, shots_all_df, home_team_df, away_team_df, master_matchreports_dict, match_reports_list, season, shots_all_df_list):
    """
    Process and save the scraped data from each match report.

    Args:
        match_report_url (str): url of the match report page
        shots_all_df (pd.DataFrame): DataFrame of all shots data
        home_team_df (pd.DataFrame): home team data
        away_team_df (pd.DataFrame): away team data
        master_matchreports_dict (dict): Dictionary of scraped match report data
        match_reports_list (list): List of all matches dataframes
        season (str): Season of the match
        shots_all_df_list (list): List of all shots dataframes

    Returns:
        tuple: Updated data structures
    """
    if home_team_df is not None and away_team_df is not None:
        single_matchreport_df = pd.concat([home_team_df, away_team_df])

    # Add 'season' column
    single_matchreport_df['season'] = season

    single_matchreport_df = cleaning_home_away_reports_tables(
        single_matchreport_df)

    # Append single_matchreport_df to the list
    match_reports_list.append(single_matchreport_df)

    # Append shots_all_df to the list
    shots_all_df_list.append(shots_all_df)

    if len(home_team_df) > 0 and len(away_team_df) > 0:
        # store match report data in master_matchreports_dict
        match_report_uuid = generate_uuid(
            home_team_df['team'].iloc[0], away_team_df['team'].iloc[0], season)
    else:
        print("Either 'home_team_df' or 'away_team_df' is empty.")
        # handle the case when the dataframes are empty

    master_matchreports_dict[match_report_uuid] = {
        'shots_all_df': shots_all_df.to_dict('records'),
        'home_team_df': home_team_df.to_dict('records'),
        'away_team_df': away_team_df.to_dict('records')
    }

    # store full_season_matchreports_df to dict
    full_season_matchreports_dict = {
        'full_season_matchreports_df': pd.concat(match_reports_list).to_dict('records')
    }

    return master_matchreports_dict, match_reports_list, full_season_matchreports_dict, shots_all_df_list

        

def scrape_all_match_reports(base_url, table_body, df, team_table_ids, season, all_shots_all_df, ):
    try:
        clear_output(wait=True)

        match_report_urls = {}

        for i, row in enumerate(table_body.find_all('tr')):
            match_report_column = row.find('td', {'data-stat': 'match_report'})
            if "spacer" in row.get("class", []):
                continue
            elif match_report_column is not None and match_report_column.find('a') is not None:
                match_report_url = match_report_column.find('a')['href']
                match_report_urls[i] = match_report_url

        print(f"Found {len(match_report_urls)} match reports")

        # Use the provided all_shots_all_df as the initial list
        shots_all_df_list = all_shots_all_df

        master_matchreports_dict = {}
        match_reports_list = []  # Initialize the list here
        full_season_matchreports_dict = {}
        full_season_matchreports_df = pd.DataFrame()

        for i, match_report_url in enumerate(match_report_urls.values()):
            try:
                shots_all_df, home_team_df, away_team_df = scrape_match_report(
                    match_report_url, base_url, team_table_ids)

                if home_team_df is not None and away_team_df is not None and shots_all_df is not None:
                    master_matchreports_dict, full_season_matchreports_df, full_season_matchreports_dict, shots_all_df_list = process_and_save_scraped_data(
                        match_report_url, shots_all_df, home_team_df, away_team_df, master_matchreports_dict, match_reports_list, season, shots_all_df_list)

                    time.sleep(3)

            except Exception as e:
                return None  # Return None in case of error.

        # Concatenate all the DataFrames at the end
        if len(match_reports_list) > 0:
            full_season_matchreports_df = pd.concat(match_reports_list)
        else:
            return None

        return master_matchreports_dict, full_season_matchreports_df, full_season_matchreports_dict, shots_all_df_list

    except Exception as e:
        return None  # Return None in case of a high-level function error.


def main():
    """
    Main function to scrape all match reports from fbref.com.

    Args:
        test (bool): if True, only scrape first 5 match reports of the 2020-2021 and 2021-2022 seasons.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()

    base_url = "https://fbref.com"
    headers_request = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
        "Accept-Encoding": "*",
        "Connection": "keep-alive",
    }

    # The underscore (_) is used to ignore the season returned from get_url_table_ids function.
    url_table_ids, _ = get_url_table_ids()

    headers = []
    dataframes = []
    master_dict = {}
    seasons_dataframes_dict = {}
    seasons_dicts_dict = {}
    all_seasons_data = []  # Initialize the list outside the loop
    all_seasons_dicts = []  # Initialize the list here
    all_shots_all_df = []  # Initialize the list here

    try:
        for url, table_id in url_table_ids.items():
            clear_output(wait=True)

            # Determine season based on URL and table_id
            if "/en/comps/9/schedule/Premier-League-Scores-and-Fixtures" in url:
                # For current_year_url format, extract season from table_id
                season = table_id.split('_')[1]
            else:
                season = url.split('/')[-1].split('-')[0] + \
                    '-' + url.split('/')[-1].split('-')[1]

            data = get_season_data(url, headers_request)

            if data is None:
                continue

            try:
                df, headers, table_body = prepare_season_dataframe(
                    data, headers, table_id, url)

            except ValueError as e:
                raise

            dataframes.append(df)

            team_table_ids = ["_summary", "_passing",
                              "_passing_types", "_defense", "_possession", "_misc"]


            master_matchreports_dict, full_season_matchreports_df, full_season_matchreports_dict, all_shots_all_df = scrape_all_match_reports(
                base_url, table_body, df, team_table_ids, season, all_shots_all_df, )

            # Update master dict
            master_dict = {**master_dict, **master_matchreports_dict}

            # Append the dictionary to the list
            all_seasons_dicts.append(full_season_matchreports_dict.copy())

            # Save dictionary with season as the key
            seasons_dicts_dict[season] = full_season_matchreports_dict.copy()

            # Copy full_season_matchreports_df and save it with the season as the key
            if 'season' not in full_season_matchreports_df.columns:
                print(
                    f"'season' column not in full_season_matchreports_df: {season}")
            else:
                seasons_dataframes_dict[season] = full_season_matchreports_df.copy(
                )

            # Append the DataFrame to the list
            all_seasons_data.append(full_season_matchreports_df)

        # concatenate dataframes in dataframes
        for results_df in dataframes:
            # make all columns lowercase
            results_df.columns = results_df.columns.str.lower()
            drop_cols = ['score', 'match_report', 'notes']
            results_df = results_df.drop(columns=[col for col in drop_cols if col in results_df])

            desired_columns_order = ['season', 'gameweek', 'home_team', 'home_xg', 'away_xg', 'away_team', 'home_score', 'away_score', 'date', 'referee', 'venue', 'dayofweek',	'start_time', 'attendance']
            rest_of_columns = [col for col in results_df.columns if col not in desired_columns_order]
            results_df = results_df.reindex(desired_columns_order + rest_of_columns, axis=1)

        # Concatenate all the shots_all_df DataFrames together
        return seasons_dataframes_dict, all_seasons_dicts, seasons_dicts_dict, master_dict

    except Exception as e:
        raise
    
if __name__ == "__main__":
    seasons_dataframes_dict, all_seasons_dicts, seasons_dicts_dict, master_dict = main()
