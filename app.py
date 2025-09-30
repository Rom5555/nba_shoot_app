import acceuil_page, data_page, deep_learning_page, models_page, visualisation_page
import streamlit as st


# Configuration de la page
st.set_page_config(page_title="NBA Shoot Prediction", layout="wide")

# Menu principal
menu = {
    "Accueil": acceuil_page.show,
    "Data": data_page.show,
    "Visualisation": visualisation_page.show,
    "Modèles": models_page.show,
    "Deep Learning": deep_learning_page.show,
    "Prédiction": None,
}

# Sélection via la sidebar
choice = st.sidebar.selectbox("Navigation", list(menu.keys()))

# Lancer la page correspondante
menu[choice]()
