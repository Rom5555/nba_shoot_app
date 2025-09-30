import pandas as pd

def load_data(filename="data/dataset_V5.csv"):
    """Charge un fichier CSV depuis le dossier data"""
    return pd.read_csv(filename)
