import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from pathlib import Path

# ----------------------
# Features utilisées
# ----------------------
FEATURES = ["SHOT_DISTANCE", "ACTION_ZONE_PCT", "DAYS_SINCE_LAST_GAME"]
TARGET = "SHOT_MADE_FLAG"

# ----------------------
# Chargement dataset
# ----------------------
def load_data(path="data/dataset_V5.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df

# ----------------------
# Création pipeline simple
# ----------------------
def create_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    return pipeline

# ----------------------
# Entraînement rapide
# ----------------------
def train_pipeline(df):
    X = df[FEATURES]
    y = df[TARGET]
    pipeline = create_pipeline()
    pipeline.fit(X, y)
    return pipeline

# ----------------------
# Sauvegarde / Chargement
# ----------------------
def save_pipeline(pipeline, path="models/xgboost_pipeline_game.pkl"):
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

def load_pipeline(path="models/xgboost_pipeline_game.pkl"):
    return joblib.load(path)

# ----------------------
# Fonction utilitaire prédiction
# ----------------------
def predict_shot(pipeline, input_df):
    """Retourne probabilité de réussite d'un tir"""
    proba = pipeline.predict_proba(input_df[FEATURES])[:, 1]
    return proba
