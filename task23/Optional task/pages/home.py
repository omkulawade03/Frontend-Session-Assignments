import streamlit as st

st.markdown("""
    <div class="main-header">
        <h1>🤖 Session 23: Machine Learning Frontend Dashboard</h1>
        <p>Interactive platform for Classification and Regression Models</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📌 Student Details")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Om Prashant Kulawade</h4>
        <p><b>College:</b> Zeal Polytechnic, Narhe, Pune</p>
        <p><b>Branch:</b> Artificial Intelligence and Machine Learning (AIML)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Assignment Details</h4>
        <p><b>Domain:</b> Artificial Intelligence and Machine Learning</p>
        <p><b>Assignment:</b> Session 23 AIML Optional Task</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🚀 Module Overview")

c1, c2 = st.columns(2)

with c1:
    st.info("### 🧪 Classification Suite")
    st.write("Predict malignant or benign tumors using the **Breast Cancer Wisconsin Dataset**.")
    st.markdown("""
    * **Logistic Regression**
    * **K-Nearest Neighbors (KNN)**
    * **Naive Bayes**
    """)
    if st.button("Explore Classification Modules"):
        st.switch_page("pages/classification.py")

with c2:
    st.success("### 📈 Regression Suite")
    st.write("Predict quantitative disease progression using the **Diabetes Dataset**.")
    st.markdown("""
    * **Linear Regression**
    * Multi-feature input tuning
    * Real-time regression curve analysis
    """)
    if st.button("Explore Regression Modules"):
        st.switch_page("pages/regression.py")