import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AIML Task 23 - Machine Learning Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Shared Styles
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4F46E5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Navigation setup
pages = {
    "Dashboard": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
        st.Page("pages/classification.py", title="Classification Suite", icon="🧪"),
        st.Page("pages/regression.py", title="Regression Suite", icon="📈"),
        st.Page("pages/performance.py", title="Model Performance Comparison", icon="📊"),
        st.Page("pages/about.py", title="About & Student Profile", icon="👨‍🎓"),
    ]
}

pg = st.navigation(pages)
pg.run()