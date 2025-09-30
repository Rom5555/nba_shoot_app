import streamlit as st
import os
import pandas as pd

def show():
    st.subheader("📊 Jeu de données")
    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    selected_file = st.selectbox("Choisir un fichier CSV :", csv_files)
    if selected_file:
        df = pd.read_csv(f"data/{selected_file}")
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        st.dataframe(df.head(20))
        st.write(df.describe())
