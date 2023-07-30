import pandas as pd
import numpy as np
import streamlit as st
import logging
import sys
import os
import time
import datetime

# Set up logging


def set_up_logs():
    """
    This function sets up the log file for the streamlit app.
    """
    # Set up the log file
    log_file_name = "streamlit_app_logs.log"
    log_file_path = os.path.join(os.getcwd(), log_file_name)
    logging.basicConfig(filename=log_file_path,
                        level=logging.DEBUG, format='%(asctime)s %(message)s')

    LOG_FILE_PATH = log_file_path

    return log_file_path, LOG_FILE_PATH

# Log the start of the app


def log_start_of_app(log_file_path):
    """
    This function logs the start of the streamlit app.
    """
    app_start_time = None
    app_end_time = None
    # log start time
    app_start_time = datetime.datetime.now()
    # Log the start of the app
    logging.info("Streamlit app started")

    return app_start_time

# Log the end of the app


def log_end_of_app(log_file_path, app_start_time):
    """
    This function logs the end of the streamlit app.
    """
    # log end time
    app_end_time = datetime.datetime.now()
    # Log the end of the app
    logging.info("Streamlit app ended")

    # log time elapsed
    time_elapsed = app_end_time - app_start_time
    # use this format --- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start --- to log time elapsed in minutes and seconds
    logging.info(
        f"{round(time_elapsed.total_seconds(), 2)} seconds have elapsed since the start")

# Log the start of the function


def log_start_of_function(function_name):
    """
    This function logs the start of the function.
    """
    function_start_time = None

    # log start time
    function_start_time = datetime.datetime.now()
    # Log the start of the function
    logging.info("Function {} started".format(function_name))

    return function_start_time


def log_function_is_still_running(function_name, function_start_time):
    """
    This function logs that the function is still running every 2.5 minutes.
    """
    # Log that the function is still running every 2.5 minutes
    if (datetime.datetime.now() - function_start_time).total_seconds() % 150 == 0:
        logging.info(f"Function {function_name} is still running")

# Log the end of the function


def log_end_of_function(function_name, function_start_time, app_start_time):
    """
    This function logs the end of the function.
    """
    # log end time
    function_end_time = datetime.datetime.now()
    # Log the end of the function
    logging.info("Function {} ended".format(function_name))

    # log time elapsed
    time_elapsed = function_end_time - function_start_time
    # use this format --- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start --- to log time elapsed in minutes and seconds
    logging.info(
        f"Function {function_name} took {round(time_elapsed.total_seconds(), 2)} seconds")

    # log time elapsed since start of app
    time_elapsed_since_start_of_app = function_end_time - app_start_time
    # use this format --- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start --- to log time elapsed in minutes and seconds
    logging.info(
        f"{round(time_elapsed_since_start_of_app.total_seconds(), 2)} seconds have elapsed since the start of the app")
    """
    This function logs the end of the function.
    """


# Log the start of the script
def log_start_of_script(script_name):
    """
    This function logs the start of the script.
    """
    script_start_time = None
    script_end_time = None

    script_start_time = datetime.datetime.now()
    # Log the start of the script
    logging.info("Script {} started".format(script_name))

    return script_start_time, script_end_time

# Log the end of the script


def log_end_of_script(script_name, script_start_time):
    """
    This function logs the end of the script.
    """
    # log end time
    script_end_time = datetime.datetime.now()

    # log time elapsed
    time_elapsed = script_end_time - script_start_time
    # Log the start of the script, use this format --- {round((time.time() - start_time) / 60, 2)} minutes, ({round(time.time() - start_time, 2)} seconds) have elapsed since the start --- to log time elapsed in minutes and seconds to log how long the script took to run and how long the app has been running
    logging.info(f"Script {script_name} started, ({round((time.time() - script_start_time) / 60, 2)} minutes, ({round(time.time() - script_start_time, 2)} seconds) have elapsed since the start)")

# log dataframe details


def log_dataframe_details(dataframe_name, dataframe):
    """
    This function logs the dataframe details.
    """
    logging.info(
        f"Dataframe {dataframe_name} has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns\ncolumns: {dataframe.columns}\nhead:\n{dataframe.head()}\ntail:\n{dataframe.tail()}")

# log specific info message


def log_specific_info_message(message):
    """
    This function logs a specific info message.
    """
    logging.info(message)


def log_dict_contents(dict, dict_name):
    """
    This function prints the contents of a dictionary.
    """
    # print dict name
    print(f"{dict_name}:")
    keys_list = list(dict.keys())
    # print len of keys list
    print(f"Number of Dict Keys: {len(keys_list)}")
    # print keys list
    print(f"List of Dict Keys:\n{keys_list}")
