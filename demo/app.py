"""Presentation app for the final project. Run from the repo root:

    .venv\\Scripts\\streamlit run demo\\app.py

Pages follow the presentation order, one per speaker.
"""
import streamlit as st

st.set_page_config(page_title="Multi-agent order picking", page_icon=":material/forklift:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/intro.py", title="1. Introduction", icon=":material/home:", default=True, url_path="intro"),
        st.Page("app_pages/problem.py", title="2. Problem formulation", icon=":material/grid_on:", url_path="problem"),
        st.Page("app_pages/method.py", title="3. Method", icon=":material/functions:", url_path="method"),
        st.Page("app_pages/results.py", title="4. Results", icon=":material/monitoring:", url_path="results"),
        st.Page("app_pages/discussion.py", title="5. Discussion and conclusion", icon=":material/lightbulb:", url_path="discussion"),
    ],
    position="sidebar",
)

with st.sidebar:
    st.caption("Multi-agent order picking in a simulated warehouse")
    st.caption("Jönköping University · Introduction to Reinforcement Learning · 2026")

page.run()
