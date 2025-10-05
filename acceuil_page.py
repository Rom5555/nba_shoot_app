import streamlit as st

def show():
    st.title("🏀 NBA Shoot Prediction App")
    st.subheader("📌 Présentation du projet")
    with open("fichiers_markdown/introduction.md", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
