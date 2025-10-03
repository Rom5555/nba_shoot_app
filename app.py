import acceuil_page, data_page, deep_learning_page, models_page, visualisation_page, prediction_page, conclusion_page
import streamlit as st
import asyncio
import warnings

# 🔹 Patch asyncio pour Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 🔹 Ignorer les warnings PyTorch liés à __path__._path
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch")


# Configuration de la page
st.set_page_config(page_title="NBA Shoot Prediction", layout="wide")

# Menu principal
menu = {
    "Accueil": acceuil_page.show,
    "Data": data_page.show,
    "Visualisation": visualisation_page.show,
    "Modèles": models_page.show,
    "Deep Learning": deep_learning_page.show,
    "Prédiction": prediction_page.show,
    "Conclusion": conclusion_page.show
}

# Sélection via la sidebar
choice = st.sidebar.selectbox("Navigation", list(menu.keys()))

# Lancer la page correspondante
menu[choice]()
