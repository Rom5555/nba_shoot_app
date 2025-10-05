import streamlit as st
import pandas as pd
from visualization import (
    plot_target_distribution, plot_shot_type_by_player, plot_action_type_heatmap,
    plot_shot_distance, plot_violin_distance, plot_last_5d_pct,
    plot_shot_zone_area, plot_all_players_heatmap_blue, plot_all_players_scatter, plot_shot_zone_range
)

@st.cache_data
def load_data(path="data/dataset_V5.csv"):
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    return df

def show():
    st.subheader("📈 Visualisation des données")
    df = load_data()
    player_list = sorted(df["PLAYER_NAME"].unique())

    plot_options = {
        "Distribution cible": plot_target_distribution,
        "Tirs par joueur et type": plot_shot_type_by_player,
        "Heatmap type d'action": plot_action_type_heatmap,
        "Couloirs de tir par joueur": plot_shot_zone_area,
        "Tranches de tir par joueur": plot_shot_zone_range,
        "Heatmap sur terrain": plot_all_players_heatmap_blue,
        "Dispersion sur terrain": plot_all_players_scatter,
        "Distance vs résultat": plot_shot_distance,
        "Violin distance": plot_violin_distance,
        "% cumulé tir sur 5 jours": plot_last_5d_pct
    }

    plot_choice = st.sidebar.selectbox("Choisir un graphique", list(plot_options.keys()))

    if plot_choice in ["Tirs par joueur et type", "Heatmap type d'action", "Violin distance",
                       "Couloirs de tir par joueur", "Heatmap sur terrain","Dispersion sur terrain",
                       "% cumulé tir sur 5 jours", "Tranches de tir par joueur"]:
        selected_players = st.sidebar.multiselect("Sélectionner les joueurs", player_list, default=player_list)

    plot_fn = plot_options[plot_choice]

    if plot_choice in ["Distribution cible", "Distance vs résultat"]:
        st.pyplot(plot_fn(df))
    elif plot_choice == "% cumulé tir sur 5 jours":
        for fig in plot_fn(df, selected_players):
            st.pyplot(fig)
    else:
        st.pyplot(plot_fn(df, selected_players))
