import streamlit as st
from dashboard import auth
from dashboard.views import aoi_inference, research_results

st.set_page_config(
    page_title="LandSentry",
    layout="wide",
    page_icon="🛰️"
)

if not st.session_state.get("logged_in", False):
    auth.login()
else:
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.get('username', 'User')}!")
        st.caption(f"Role: {st.session_state.get('role', 'Unknown')}")

        pages = {
            "AOI Analysis":       aoi_inference.render_page,
            "Research Results":   research_results.render_page,
        }

        selection = st.radio("Navigation", list(pages.keys()))

        st.divider()
        if st.button("Logout"):
            auth.logout()

    if selection and selection in pages:
        pages[selection]()