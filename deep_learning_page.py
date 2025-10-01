import streamlit as st
import pandas as pd
import os
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
import deep_learning as deep  # ton module deep_learning.py

# ---------------- Cache Data ----------------
@st.cache_data
def load_data_deep(path="data/dataset_V5.csv"):
    df = pd.read_csv(path)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df

# ---------------- Helpers ----------------
def save_model_light(path, obj, tabnet_model=None, tabnet_path=None):
    """
    Sauvegarde compressée du modèle + préprocesseur + history.
    Si TabNet, le modèle est sauvegardé séparément pour éviter les problèmes de pickling.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        joblib.dump(obj, path, compress=9)
        return True, None
    except Exception:
        if tabnet_model and tabnet_path:
            try:
                tabnet_model.save_model(tabnet_path)
            except Exception:
                pass
            obj_copy = obj.copy()
            if "model" in obj_copy:
                obj_copy["model"] = None
                obj_copy["_tabnet_model_path"] = tabnet_path
            try:
                joblib.dump(obj_copy, path, compress=9)
                return True, f"model_saved_separately:{tabnet_path}"
            except Exception as e2:
                return False, str(e2)
        return False, "Échec de sauvegarde"

def load_model_light(path):
    """
    Charge l'objet sauvegardé. Si _tabnet_model_path existe, recharge le modèle TabNet.
    """
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and loaded.get("_tabnet_model_path"):
        model_path = loaded["_tabnet_model_path"]
        tab_model = TabNetClassifier()
        try:
            tab_model.load_model(model_path)
            loaded["model"] = tab_model
        except Exception:
            loaded["model"] = None
    return loaded

# ---------------- Page Deep Learning ----------------
def show():
    st.subheader("🧠 Deep Learning")

    df = load_data_deep()
    drop_vars = ['SHOT_MADE_FLAG', 'GAME_DATE', 'GAME_EVENT_ID']
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    split_index = int(0.8 * len(df))
    split_date = df['GAME_DATE'].iloc[split_index]

    X_train = df[df['GAME_DATE'] <= split_date].drop(columns=[c for c in drop_vars if c in df.columns])
    y_train = df[df['GAME_DATE'] <= split_date]['SHOT_MADE_FLAG']
    X_test = df[df['GAME_DATE'] > split_date].drop(columns=[c for c in drop_vars if c in df.columns])
    y_test = df[df['GAME_DATE'] > split_date]['SHOT_MADE_FLAG']

    deep_option = st.selectbox("Choisir le modèle Deep", ["Keras MLP", "TabNet"])
    deep_name_lower = deep_option.replace(" ", "_").lower()
    saved_models = {
        "standard": f"deep_models/{deep_name_lower}_standard.pkl",
        "tuned": f"deep_models/{deep_name_lower}_tuned.pkl"
    }

    tab1, tab2, tab3 = st.tabs(["📂 Charger", "🔹 Simple", "🔧 Tuning"])

    # ---------------- Tab1 : Charger ----------------
    with tab1:
        available_models = {k: v for k, v in saved_models.items() if os.path.exists(v)}
        if available_models:
            model_to_load = st.radio("Modèles disponibles :", list(available_models.keys()))
            if st.button("Charger", key="load_deep_model"):
                try:
                    loaded = load_model_light(available_models[model_to_load])
                except Exception as e:
                    st.error(f"Erreur au chargement : {e}")
                    loaded = None

                if loaded:
                    st.success(f"✅ {model_to_load} chargé")

                    preproc = loaded.get("preprocessor") if isinstance(loaded, dict) else None
                    X_test_proc = (preproc.transform(X_test).astype('float32')
                                   if preproc else X_test.fillna(0).astype('float32'))

                    history_loaded = loaded.get("history") if isinstance(loaded, dict) else None
                    model_obj = loaded.get("model") if isinstance(loaded, dict) else loaded

                    if model_obj is None:
                        st.warning("Le modèle n'a pas pu être reconstitué. Si c'est un TabNet, vérifier le fichier séparé de sauvegarde.")
                    else:
                        deep.eval_deep(model_obj, X_test_proc, y_test,
                                       history=history_loaded, plot_cm=True, plot_loss=True, show_on_streamlit=True)
        else:
            st.info("Aucun modèle sauvegardé trouvé.")

    # ---------------- Tab2 : Entraînement simple ----------------
    with tab2:
        IS_CLOUD = st.runtime.scriptrunner.is_running_with_streamlit
        
        if st.button("Entraîner et évaluer", key="train_deep_simple"):
            if IS_CLOUD:
                st.warning("🚫 Le tuning est désactivé sur Streamlit Cloud (trop lourd).")
                st.info("👉 Charge un modèle déjà entraîné depuis l’onglet 📂 Charger.")
            else:           
                numerical_continuous = deep.numerical_continuous
                numerical_counts = deep.numerical_counts
                categorical = deep.categorical
                boolean = deep.boolean

                X_train_proc, X_test_proc, preprocessor = deep.preprocess_data(
                    X_train, X_test,
                    numerical_continuous, numerical_counts, categorical, boolean
                )

                with st.spinner(f"Entraînement {deep_option} en cours..."):
                    model_type = 'keras' if deep_option == "Keras MLP" else 'tabnet'
                    model, report, auc_val, fig_cm, fig_loss, history = deep.train_and_eval(
                        X_train_proc, y_train, X_test_proc, y_test,
                        model_type=model_type, epochs=50, batch_size=1024
                    )

                deep.eval_deep(model, X_test_proc, y_test, history=history,
                            plot_cm=True, plot_loss=True, show_on_streamlit=True)

                # ---------------- Sauvegarde allégée ----------------
                os.makedirs("deep_models", exist_ok=True)
                save_path = f"deep_models/{deep_name_lower}_standard.pkl"
                save_obj = {"model": model, "preprocessor": preprocessor, "history": history}

                if deep_option == "TabNet":
                    tabnet_model_path = f"deep_models/{deep_name_lower}_model.zip"
                    ok, info = save_model_light(save_path, save_obj, tabnet_model=model, tabnet_path=tabnet_model_path)
                else:
                    ok, info = save_model_light(save_path, save_obj)

                if ok:
                    st.success(f"✅ {deep_option} entraîné et sauvegardé.")
                    if info:
                        st.info(f"Info sauvegarde: {info}")
                else:
                    st.error(f"❌ Échec de sauvegarde : {info}")

    # ---------------- Tab3 : Tuning ----------------
    with tab3:
        st.info("Tuning non implémenté ici. Utiliser optuna / GridSearch dans le module deep_learning si nécessaire.")
