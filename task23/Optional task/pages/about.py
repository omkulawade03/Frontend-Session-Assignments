import streamlit as st

st.header("👨‍🎓 About & Student Profile")

col_profile, col_details = st.columns([1, 2])

with col_profile:
    # Avatar/Profile placeholder
    st.markdown("""
    <div style="text-align: center; padding: 1rem; border: 2px solid #4F46E5; border-radius: 12px;">
        <h2>👤</h2>
        <h3>Om Prashant Kulawade</h3>
        <p>AIML Student</p>
    </div>
    """, unsafe_allow_html=True)

with col_details:
    st.markdown("""
    ### Student Information
    * **Name:** Om Prashant Kulawade
    * **College:** Zeal Polytechnic, Narhe, Pune
    * **Branch:** Artificial Intelligence and Machine Learning (AIML)
    * **Domain:** Artificial Intelligence and Machine Learning
    * **Project:** Session 23 AIML Dashboard
    
    ### Technologies Used
    `Python` `Streamlit` `Pandas` `NumPy` `Scikit-learn` `Matplotlib`
    """)

st.markdown("---")

st.markdown("""
### 📌 Project Description & Scope
This Streamlit Dashboard was designed as part of **Session 23 Optional Task** for the AIML domain. It provides an intuitive, web-based interface for evaluating standard Supervised Machine Learning algorithms across classification and regression domains.

#### Future Scope
* Integration of real-time hyperparameter tuning directly from sidebar controls.
* Support for custom dataset upload via CSV drag-and-drop.
* Automated exportable PDF evaluation reports for predictions.

#### Acknowledgement
Special thanks to the faculty and mentors at **Zeal Polytechnic, Narhe, Pune** for guidance throughout the Machine Learning curriculum.

---
**Contact & Repositories:**  
* 🐙 **GitHub:** [github.com/placeholder](https://github.com/omkulawade03/Frontend-Session-Assignments/tree/main/task23/Optional%20task)  
* 📧 **Email:** [student@example.com](Omkulawade534@gmail.com)
""")