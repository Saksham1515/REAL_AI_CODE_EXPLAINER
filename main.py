import streamlit as st

pg = st.navigation([
    st.Page("Code_Explainer.py", title="Code Explainer",icon="🤖"),
    st.Page("RAG_QandA.py", title="RAG Q&A", icon="👨‍💻")
])
pg.run()