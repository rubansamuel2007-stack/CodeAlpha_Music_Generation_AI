import streamlit as st

st.set_page_config(page_title="Test App")

st.title("🎵 MelodyMind AI")
st.subheader("AI Music Generation System")

st.success("Streamlit is working successfully!")

st.write("Welcome to your AI Music Generator.")

name = st.text_input("Enter your name")

if name:
    st.write(f"Welcome, {name}!")