# app.py
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib


from visualization import (
    plot_target_distribution, plot_shot_type_by_player, plot_action_type_heatmap,
    plot_shot_distance, plot_violin_distance, plot_last_5d_pct,
    plot_shot_zone_area, plot_all_players_scatter, plot_shot_zone_range
)

from models import (
    get_preprocessor, create_pipeline, train_and_eval,
    grid_search_pipeline, randomized_search_pipeline, optuna_tune, evaluate_model
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from pytorch_tabnet.tab_model import TabNetClassifier 
import numpy as np
import deep_learning as deep  # ton module deep_learning.py corrigé



# -------------------------------
# Configuration Streamlit
# -------------------------------
st.set_page_config(page_title="NBA Shoot Prediction", layout="wide")
st.title("🏀 NBA Shoot Prediction App")
st.write("Bienvenue dans ton application Streamlit !")

# -------------------------------
# Menu principal
# -------------------------------
menu = ["Accueil", "Data", "Visualisation", "Modèles", "Deep Learning", "Prédiction"]
choice = st.sidebar.selectbox("Navigation", menu)

# -------------------------------
# Accueil
# -------------------------------
if choice == "Accueil":
    st.subheader("📌 Présentation du projet")
    st.write("Projet de prédiction de réussite/échec des shoots NBA.")

# -------------------------------
# Data
# -------------------------------
elif choice == "Data":
    st.subheader("📊 Jeu de données")
    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    selected_file = st.selectbox("Choisir un fichier CSV :", csv_files)
    if selected_file:
        df = pd.read_csv(f"data/{selected_file}")
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        st.dataframe(df.head(20))
        st.write(df.describe())

# -------------------------------
# Visualisation
# -------------------------------
elif choice == "Visualisation":
    st.subheader("📈 Visualisation des données")

    @st.cache_data
    def load_data(path="data/dataset_V5.csv"):
        df = pd.read_csv(path, parse_dates=["GAME_DATE"])
        return df

    df = load_data()
    player_list = sorted(df["PLAYER_NAME"].unique())
    plot_options = {
        "Distribution cible": plot_target_distribution,
        "Tirs par joueur et type": plot_shot_type_by_player,
        "Heatmap type de tir": plot_action_type_heatmap,
        "Zones de tir par joueur": plot_shot_zone_area,
        "Distance de tir par joueur": plot_shot_zone_range,
        "Dispersion sur terrain": plot_all_players_scatter,
        "Distance vs résultat": plot_shot_distance,
        "Violin distance": plot_violin_distance,
        "% cumulé tir par tir": plot_last_5d_pct
    }

    plot_choice = st.sidebar.selectbox("Choisir un graphique", list(plot_options.keys()))
    if plot_choice in ["Tirs par joueur et type", "Heatmap type de tir", "Violin distance",
                       "Zones de tir par joueur", "Dispersion sur terrain",
                       "% cumulé tir par tir", "Distance de tir par joueur"]:
        selected_players = st.sidebar.multiselect("Sélectionner les joueurs", player_list, default=player_list)

    plot_fn = plot_options[plot_choice]
    if plot_choice in ["Distribution cible", "Distance vs résultat"]:
        st.pyplot(plot_fn(df))
    elif plot_choice == "% cumulé tir par tir":
        for fig in plot_fn(df, selected_players):
            st.pyplot(fig)
    else:
        st.pyplot(plot_fn(df, selected_players))

# ------------------------------- 
# Modèles
# -------------------------------
elif choice == "Modèles":
    st.subheader("🤖 Modèles")

    # -------------------------------
    # Chargement des données
    @st.cache_data
    def load_data(path="data/dataset_V5.csv"):
        df = pd.read_csv(path)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df

    df = load_data()
    drop_vars = ['SHOT_MADE_FLAG', 'GAME_DATE', 'GAME_EVENT_ID']
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    split_index = int(0.8 * len(df))
    split_date = df['GAME_DATE'].iloc[split_index]

    X_train = df[df['GAME_DATE'] <= split_date].drop(columns=drop_vars)
    y_train = df[df['GAME_DATE'] <= split_date]['SHOT_MADE_FLAG']
    X_test = df[df['GAME_DATE'] > split_date].drop(columns=drop_vars)
    y_test = df[df['GAME_DATE'] > split_date]['SHOT_MADE_FLAG']

    # Choix du modèle
    model_option = st.selectbox("Choisir le modèle", ["Random Forest", "XGBoost"])
    model_name_lower = model_option.replace(" ", "_").lower()
    saved_models = {
        "standard": f"models/{model_name_lower}_pipeline.pkl",
        "grid": f"models/{model_name_lower}_pipeline_grid.pkl",
        "randomized": f"models/{model_name_lower}_pipeline_randomized.pkl",
        "optuna": f"models/{model_name_lower}_pipeline_optuna.pkl"
    }

    # -------------------------------
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📂 Charger", "🔹 Simple", "🔧 Tuning"])

    # 📂 Charger un modèle
    with tab1:
        available_models = {k: v for k, v in saved_models.items() if os.path.exists(v)}
        if available_models:
            model_to_load = st.radio("Modèles disponibles :", list(available_models.keys()))
            if st.button("Charger", key="load_model"):
                loaded_model = joblib.load(available_models[model_to_load])
                st.success(f"✅ {model_to_load} chargé")

                # Figure d'importances si dispo
                fig_path = available_models[model_to_load].replace(".pkl", "_fig_imp.pkl")
                fig_imp = joblib.load(fig_path) if os.path.exists(fig_path) else None

                evaluate_model(
                    loaded_model, X_test, y_test,
                    model_name=f"{model_option} ({model_to_load})",
                    feature_importances=(fig_imp is not None),
                    fig_importance=fig_imp
                )
        else:
            st.info("Aucun modèle sauvegardé trouvé.")

    # 🔹 Entraîner un modèle simple
    with tab2:
        if st.button("Entraîner et évaluer", key="train_simple"):
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1) \
                if model_option == "Random Forest" else \
                XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)

            pipeline, report, auc, fig_cm = train_and_eval(base_model, X_train, y_train, X_test, y_test)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("AUC ROC", f"{auc:.4f}")
                st.pyplot(fig_cm)
            with col2:
                st.write("### Rapport (simplifié)")
                st.dataframe(pd.DataFrame(report).T[['precision','recall','f1-score']])

            os.makedirs("models", exist_ok=True)
            joblib.dump(pipeline, saved_models["standard"])
            st.info(f"💾 Sauvegardé dans {saved_models['standard']}")

    # 🔧 Hyperparameter Tuning
    with tab3:
        tuning_method = st.radio("Méthode :", ["GridSearch", "RandomizedSearch", "Optuna"])
        key_map = {"GridSearch": "grid", "RandomizedSearch": "randomized", "Optuna": "optuna"}

        if st.button("Lancer le tuning", key="run_tuning"):
            fig_imp = None

            # Définition modèle + paramètres
            if model_option == "Random Forest":
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'classifier__n_estimators': [200, 400, 600],
                    'classifier__max_depth': [5, 10, 15],
                    'classifier__min_samples_split': [2, 5, 10]
                }
                param_space = lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                    "max_depth": trial.suggest_int("max_depth", 5, 15),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "random_state": 42, "n_jobs": -1
                }
            else:
                base_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.05, 0.1],
                    'classifier__subsample': [0.6, 0.8, 1.0],
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                    'classifier__gamma': [0, 1, 5]
                }
                param_space = lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "random_state": 42, "n_jobs": -1, "eval_metric": "logloss",
                    "use_label_encoder": False
                }

            with st.spinner(f"⏳ {tuning_method} en cours..."):
                if tuning_method == "GridSearch":
                    grid, fig_imp, _ = grid_search_pipeline(base_model, X_train, y_train, X_test, y_test, param_grid)
                    best_pipeline = grid.best_estimator_
                elif tuning_method == "RandomizedSearch":
                    search = randomized_search_pipeline(base_model, X_train, y_train, param_grid, n_iter=20, cv=3)
                    best_pipeline = search.best_estimator_
                else:
                    best_pipeline = optuna_tune(type(base_model), param_space, X_train, y_train, n_trials=20, scoring="f1")

                evaluate_model(
                    best_pipeline, X_test, y_test,
                    model_name=f"{model_option} ({tuning_method})",
                    feature_importances=(fig_imp is not None),
                    fig_importance=fig_imp
                )

                # Sauvegarde
                os.makedirs("models", exist_ok=True)
                save_path = f"models/{model_name_lower}_pipeline_{key_map[tuning_method]}.pkl"
                joblib.dump(best_pipeline, save_path)
                if fig_imp is not None:
                    joblib.dump(fig_imp, save_path.replace(".pkl", "_fig_imp.pkl"))
                st.info(f"💾 Sauvegardé dans {save_path}")


# ----------------------------
# Deep Learning Section
# ----------------------------


elif choice == "Deep Learning":
    
    st.subheader("🧠 Deep Learning")

    # --- Chargement des données
    @st.cache_data
    def load_data_deep(path="data/dataset_V5.csv"):
        df = pd.read_csv(path)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        return df

    df = load_data_deep()
    drop_vars = ['SHOT_MADE_FLAG', 'GAME_DATE', 'GAME_EVENT_ID']
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    # split temporel 80/20
    split_index = int(0.8 * len(df))
    split_date = df['GAME_DATE'].iloc[split_index]

    X_train = df[df['GAME_DATE'] <= split_date].drop(columns=[c for c in drop_vars if c in df.columns])
    y_train = df[df['GAME_DATE'] <= split_date]['SHOT_MADE_FLAG']
    X_test = df[df['GAME_DATE'] > split_date].drop(columns=[c for c in drop_vars if c in df.columns])
    y_test = df[df['GAME_DATE'] > split_date]['SHOT_MADE_FLAG']

    # --- Choix du modèle
    deep_option = st.selectbox("Choisir le modèle Deep", ["Keras MLP", "TabNet"])
    deep_name_lower = deep_option.replace(" ", "_").lower()
    saved_models = {
        "standard": f"deep_models/{deep_name_lower}_standard.pkl",
        "tuned": f"deep_models/{deep_name_lower}_tuned.pkl"
    }

    tab1, tab2, tab3 = st.tabs(["📂 Charger", "🔹 Simple", "🔧 Tuning"])

    # ----- Helpers -----
    def save_model_obj(path, obj, tabnet_model=None, tabnet_path=None):
        """
        Essaie de joblib.dump directement. Si échec (ex: TabNet non picklable), sauvegarde
        l'objet partiel et, si fourni, le modèle TabNet séparément via save_model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            joblib.dump(obj, path)
            return True, None
        except Exception as e:
            # si TabNet fourni, sauvegarde model.save_model et sauve dict sans model
            if tabnet_model is not None and tabnet_path is not None:
                try:
                    tabnet_model.save_model(tabnet_path)
                except Exception:
                    pass
                # remove model from obj and save remaining
                obj_copy = obj.copy()
                if "model" in obj_copy:
                    obj_copy["model"] = None
                    obj_copy["_tabnet_model_path"] = tabnet_path
                try:
                    joblib.dump(obj_copy, path)
                    return True, f"model_saved_separately:{tabnet_path}"
                except Exception as e2:
                    return False, str(e2)
            return False, str(e)

    def load_model_obj(path):
        """
        Charge via joblib. Si l'objet indique _tabnet_model_path, on recharge le modèle TabNet depuis le chemin.
        """
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and loaded.get("_tabnet_model_path"):
            model_path = loaded["_tabnet_model_path"]
            # charger un nouvel estimator TabNet et charger les poids
            tab_model = TabNetClassifier()
            try:
                tab_model.load_model(model_path)
                loaded["model"] = tab_model
            except Exception:
                # fallback: laisser model à None et informer l'utilisateur
                loaded["model"] = None
        return loaded

    # ----- Tab1 : Charger -----
    with tab1:
        available_models = {k: v for k, v in saved_models.items() if os.path.exists(v)}
        if available_models:
            model_to_load = st.radio("Modèles disponibles :", list(available_models.keys()))
            if st.button("Charger", key="load_deep_model"):
                try:
                    loaded = load_model_obj(available_models[model_to_load])
                except Exception as e:
                    st.error(f"Erreur au chargement : {e}")
                    loaded = None

                if loaded is None:
                    st.error("Impossible de charger le modèle.")
                else:
                    st.success(f"✅ {model_to_load} chargé")

                    # Préprocessing pour test
                    if isinstance(loaded, dict) and "preprocessor" in loaded and loaded["preprocessor"] is not None:
                        preproc = loaded["preprocessor"]
                        try:
                            X_test_proc = preproc.transform(X_test).astype('float32')
                        except Exception:
                            X_test_proc = X_test.fillna(0).astype('float32').values
                    else:
                        X_test_proc = X_test.fillna(0).astype('float32')

                    # Récupérer history si présent
                    history_loaded = None
                    if isinstance(loaded, dict) and "history" in loaded:
                        history_loaded = loaded["history"]

                    # Récupérer modèle
                    model_obj = None
                    if isinstance(loaded, dict) and "model" in loaded:
                        model_obj = loaded["model"]
                    else:
                        model_obj = loaded

                    if model_obj is None:
                        st.warning("Le modèle n'a pas pu être reconstitué. Si c'est un TabNet, vérifier le fichier séparé de sauvegarde.")
                    else:
                        # Évaluer et afficher (eval_deep gère Keras et TabNet)
                        deep.eval_deep(model_obj, X_test_proc, y_test, history=history_loaded, plot_cm=True, plot_loss=True, show_on_streamlit=True)
        else:
            st.info("Aucun modèle sauvegardé trouvé.")

    # ----- Tab2 : Entraînement simple -----
    with tab2:
        if st.button("Entraîner et évaluer", key="train_deep_simple"):
            numerical_continuous = deep.numerical_continuous
            numerical_counts = deep.numerical_counts
            categorical = deep.categorical
            boolean = deep.boolean

            # Préprocessing
            X_train_proc, X_test_proc, preprocessor = deep.preprocess_data(
                X_train, X_test,
                numerical_continuous, numerical_counts, categorical, boolean
            )

            # Lancer entraînement
            with st.spinner(f"Entraînement {deep_option} en cours..."):
                if deep_option == "Keras MLP":
                    model, report, auc_val, fig_cm, fig_loss, history = deep.train_and_eval(
                        X_train_proc, y_train, X_test_proc, y_test, model_type='keras', epochs=50, batch_size=1024
                    )
                else:  # TabNet
                    model, report, auc_val, fig_cm, fig_loss, history = deep.train_and_eval(
                        X_train_proc, y_train, X_test_proc, y_test, model_type='tabnet', epochs=50, batch_size=1024
                    )

            # Affichage (deep.train_and_eval n'affiche pas automatiquement)
            deep.eval_deep(model, X_test_proc, y_test, history=history, plot_cm=True, plot_loss=True, show_on_streamlit=True)

            # Sauvegarde robuste
            os.makedirs("deep_models", exist_ok=True)
            save_path = f"deep_models/{deep_name_lower}_standard.pkl"
            save_obj = {"model": model, "preprocessor": preprocessor, "history": history}

            if deep_option == "TabNet":
                # TabNet peut poser problème à pickle ; tenter joblib, sinon sauvegarder modèle séparément
                tabnet_model = model
                tabnet_model_path = f"deep_models/{deep_name_lower}_model.zip"
                ok, info = save_model_obj(save_path, save_obj, tabnet_model=tabnet_model, tabnet_path=tabnet_model_path)
            else:
                ok, info = save_model_obj(save_path, save_obj)

            if ok:
                st.success(f"✅ {deep_option} entraîné et sauvegardé.")
                if info:
                    st.info(f"Info sauvegarde: {info}")
            else:
                st.error(f"❌ Échec de sauvegarde : {info}")

    # ----- Tab3 : Tuning (esquisse) -----
    with tab3:
        st.info("Tuning non implémenté ici. Utiliser optuna / GridSearch dans le module deep_learning si nécessaire.")






# -------------------------------
# Prédiction
# -------------------------------
elif choice == "Prédiction":
    st.subheader("🎯 Prédiction en direct")
    st.write("Ici tu pourras ajouter ton terrain de basket interactif et tester des shoots.")
