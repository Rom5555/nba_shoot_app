import streamlit as st

def show():
    with open("fichiers_markdown/Rapport Final.md", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
