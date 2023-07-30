import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    list_of_files = ['specific-csvs/fdr-pts-vs.csv', 'specific-csvs/fdr_best.csv']

    # load the dataframes
    df_fdr_pts_vs = pd.read_csv(list_of_files[0])
    df_fdr_best = pd.read_csv(list_of_files[1])

    # drop unnecessary columns
    df_fdr_pts_vs = df_fdr_pts_vs.drop(['Rank', 'Sum'], axis=1)
    df_fdr_best = df_fdr_best.drop('Code', axis=1)

    # sort and set index
    df_fdr_pts_vs = df_fdr_pts_vs.sort_values('Team').set_index('Team')
    df_fdr_best = df_fdr_best.sort_values('Team').set_index('Team')

    return df_fdr_pts_vs, df_fdr_best

def clean_pts_vs(df_fdr_pts_vs):
    df_fdr_xfpts = df_fdr_pts_vs.copy()

    for col in df_fdr_xfpts.columns:
        df_fdr_xfpts[col] = df_fdr_xfpts[col].str.extract(r'\((.*?)\)', expand=False).astype(float)

    return df_fdr_xfpts

def colorcode_pts_vs(df_fdr_xfpts):
    df_fdr_color = df_fdr_xfpts.copy()

    for col in df_fdr_xfpts.columns:
        for index, row in df_fdr_xfpts.iterrows():
            if row[col] == df_fdr_xfpts[col].max():
                df_fdr_color.loc[index, col] = 'background-color: #00FF00'
            else:
                df_fdr_color.loc[index, col] = 'background-color: #FF0000'

    return df_fdr_color

def show_tables(df_fdr_pts_vs, df_fdr_best):
    gameweek = st.slider('Gameweek', 1, 38, 6)

    df_fdr_pts_vs = df_fdr_pts_vs.iloc[:, :gameweek + 1]
    df_fdr_best = df_fdr_best.iloc[:, :gameweek + 1]

    st.markdown('### xFPts Points vs')
    st.markdown('#### Gameweek(s): ' + str(gameweek))

    st.write(df_fdr_pts_vs)
    st.write(df_fdr_best)

def main():
    df_fdr_pts_vs, df_fdr_best = load_data()

    df_fdr_pts_vs = clean_pts_vs(df_fdr_pts_vs)
    df_fdr_color = colorcode_pts_vs(df_fdr_pts_vs)

    df_fdr_pts_vs_styled = df_fdr_pts_vs.style.apply(lambda _: df_fdr_color, axis=None).format("{:.2f}")

    show_tables(df_fdr_pts_vs_styled, df_fdr_best)


if __name__ == "__main__":
    main()
