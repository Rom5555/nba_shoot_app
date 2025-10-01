import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from models import (
    train_and_eval,
    grid_search_pipeline,
    randomized_search_pipeline,
    optuna_tune,
    evaluate_model
)

# ---------------- Cache Data ----------------
@st.cache_data
def load_data(path="data/dataset_V5.csv"):
    df = pd.read_csv(path)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df

# ---------------- Helpers ----------------
def save_pipeline_light(path, pipeline, fig_imp=None):
    """
    Sauvegarde pipeline compressé et léger + figure d'importance séparée si besoin.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Sauvegarde compressée pipeline
    joblib.dump(pipeline, path, compress=9)
    # Figure séparée
    if fig_imp is not None:
        joblib.dump(fig_imp, path.replace(".pkl", "_fig_imp.pkl"))

def load_pipeline_light(path):
    pipeline = joblib.load(path)
    fig_path = path.replace(".pkl", "_fig_imp.pkl")
    fig_imp = joblib.load(fig_path) if os.path.exists(fig_path) else None
    return pipeline, fig_imp

# ---------------- Page ML ----------------
def show():
    st.subheader("🤖 Modèles")

    df = load_data()
    drop_vars = ['SHOT_MADE_FLAG', 'GAME_DATE', 'GAME_EVENT_ID']
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    split_index = int(0.8 * len(df))
    split_date = df['GAME_DATE'].iloc[split_index]

    X_train = df[df['GAME_DATE'] <= split_date].drop(columns=drop_vars)
    y_train = df[df['GAME_DATE'] <= split_date]['SHOT_MADE_FLAG']
    X_test = df[df['GAME_DATE'] > split_date].drop(columns=drop_vars)
    y_test = df[df['GAME_DATE'] > split_date]['SHOT_MADE_FLAG']

    model_option = st.selectbox("Choisir le modèle", ["Random Forest", "XGBoost"])
    model_name_lower = model_option.replace(" ", "_").lower()
    saved_models = {
        "standard": f"models/{model_name_lower}_pipeline.pkl",
        "grid": f"models/{model_name_lower}_pipeline_grid.pkl",
        "randomized": f"models/{model_name_lower}_pipeline_randomized.pkl",
        "optuna": f"models/{model_name_lower}_pipeline_optuna.pkl"
    }

    tab1, tab2, tab3 = st.tabs(["📂 Charger", "🔹 Simple", "🔧 Tuning"])

    # ---------------- Tab 1 : Charger ----------------
    with tab1:
        available_models = {k: v for k, v in saved_models.items() if os.path.exists(v)}
        if available_models:
            model_to_load = st.radio("Modèles disponibles :", list(available_models.keys()))
            if st.button("Charger", key="load_model"):
                loaded_pipeline, fig_imp = load_pipeline_light(available_models[model_to_load])
                st.success(f"✅ {model_to_load} chargé")
                evaluate_model(
                    loaded_pipeline, X_test, y_test,
                    model_name=f"{model_option} ({model_to_load})",
                    feature_importances=(fig_imp is not None),
                    fig_importance=fig_imp
                )
        else:
            st.info("Aucun modèle sauvegardé trouvé.")

    # ---------------- Tab 2 : Entraînement simple ----------------
    with tab2:
        if st.button("Entraîner et évaluer", key="train_simple"):
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200, max_depth=10) \
                if model_option == "Random Forest" else \
                XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, n_estimators=100, max_depth=5)

            pipeline, report, auc, fig_cm = train_and_eval(base_model, X_train, y_train, X_test, y_test)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("AUC ROC", f"{auc:.4f}")
                st.pyplot(fig_cm)
            with col2:
                st.write("### Rapport (simplifié)")
                st.dataframe(pd.DataFrame(report).T[['precision','recall','f1-score']])

            save_pipeline_light(saved_models["standard"], pipeline)
            st.info(f"💾 Modèle compressé sauvegardé dans {saved_models['standard']}")

    # ---------------- Tab 3 : Tuning ----------------
    with tab3:
        tuning_method = st.radio("Méthode :", ["GridSearch", "RandomizedSearch", "Optuna"])
        key_map = {"GridSearch": "grid", "RandomizedSearch": "randomized", "Optuna": "optuna"}

        if st.button("Lancer le tuning", key="run_tuning"):
            # Paramètres simplifiés pour tuning rapide
            if model_option == "Random Forest":
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                param_grid = {
                    'classifier__n_estimators': [200, 400],
                    'classifier__max_depth': [5, 10],
                    'classifier__min_samples_split': [2, 5]
                }
            else:
                base_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
                param_grid = {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 5],
                    'classifier__learning_rate': [0.05, 0.1],
                    'classifier__subsample': [0.8, 1.0],
                    'classifier__colsample_bytree': [0.8, 1.0],
                    'classifier__gamma': [0, 1]
                }

            with st.spinner(f"⏳ {tuning_method} en cours..."):
                if tuning_method == "GridSearch":
                    best_pipeline, importance_df, fig_imp, report, auc, fig_cm = grid_search_pipeline(
                        base_model, X_train, y_train, X_test, y_test, param_grid
                    )
                elif tuning_method == "RandomizedSearch":
                    best_pipeline, importance_df, fig_imp, report, auc, fig_cm = randomized_search_pipeline(
                        base_model, X_train, y_train, X_test, y_test, param_grid, n_iter=20, cv=3
                    )
                else:
                    param_space = lambda trial: {
                        k.replace('classifier__', ''): trial.suggest_int(
                            k.replace('classifier__', ''), min(param_grid[k]), max(param_grid[k])
                        ) for k in param_grid
                    }
                    best_pipeline, importance_df, fig_imp, report, auc, fig_cm = optuna_tune(
                        type(base_model), param_space, X_train, y_train, X_test, y_test, n_trials=20, scoring="f1"
                    )

                evaluate_model(
                    best_pipeline, X_test, y_test,
                    model_name=f"{model_option} ({tuning_method})",
                    feature_importances=(fig_imp is not None),
                    fig_importance=fig_imp
                )

                save_path = f"models/{model_name_lower}_pipeline_{key_map[tuning_method]}.pkl"
                save_pipeline_light(save_path, best_pipeline, fig_imp)
                st.info(f"💾 Modèle compressé sauvegardé dans {save_path}")
