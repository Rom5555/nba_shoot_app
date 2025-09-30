import streamlit as st
import pandas as pd
import numpy as np
import prediction

FEATURES = ["SHOT_DISTANCE", "ACTION_ZONE_PCT", "DAYS_SINCE_LAST_GAME"]

def show():
    st.title("🏀 Mini-jeu de tir stylé avec ML")

    # --------- Chargement pipeline ----------
    @st.cache_resource
    def load_pipeline_cached():
        model_path = "models/xgboost_pipeline_game.pkl"
        try:
            return prediction.load_pipeline(model_path)
        except:
            st.warning("Modèle non trouvé. Entraînement rapide...")
            df = prediction.load_data()
            pipeline = prediction.train_pipeline(df)
            prediction.save_pipeline(pipeline, model_path)
            return pipeline

    pipeline = load_pipeline_cached()

    # --------- Chargement dataset ----------
    @st.cache_data
    def load_data(path="data/dataset_V5.csv"):
        df = pd.read_csv(path)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        return df

    df = load_data()
    players = df['PLAYER_NAME'].unique()

    # --------- Layout principal ----------
    col1, col2 = st.columns([3, 2])

    # --------- Colonne gauche (réglages + tableau) ----------
    with col1:
        player = st.selectbox("Choisir un joueur", players)
        st.markdown("**Réglage des features**")
        selected_features = {}
        for feat in FEATURES:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            default_val = float(df[df['PLAYER_NAME'] == player][feat].median())
            if feat == 'ACTION_ZONE_PCT':
                selected_features[feat] = st.slider(feat, 0.0, 100.0, default_val)
            else:
                selected_features[feat] = st.slider(feat, min_val, max_val, default_val)

        simulate_by_proba = st.checkbox("Simuler selon la probabilité", value=True)

        # Tableau aperçu juste en dessous des réglages
        st.markdown("**Aperçu entrée**")
        input_df = pd.DataFrame({k: [v] for k, v in selected_features.items()})
        st.dataframe(input_df, use_container_width=True)

    # --------- Colonne droite (résultat + bouton) ----------
    with col2:
        if "last_shot" in st.session_state:
            proba = st.session_state["last_shot"]["proba"]
            success = st.session_state["last_shot"]["success"]

            st.markdown(f"**Probabilité de réussite : {proba:.2f}**")
            if success:
                st.success("🎉 Panier marqué !")
                st.image("assets/lebron-james.gif", use_container_width=True)
            else:
                st.error("❌ Raté !")
                st.image("assets/eyebrowfinal.gif", use_container_width=True)

        st.divider()
        if st.button("Tirer ! 🎯"):
            input_df_f = input_df[FEATURES].astype(float)
            proba = float(prediction.predict_shot(pipeline, input_df_f)[0])
            success = np.random.rand() < proba if simulate_by_proba else proba >= 0.5

            st.session_state["last_shot"] = {
                "proba": proba,
                "success": success
            }
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()


if __name__ == "__main__":
    show()
