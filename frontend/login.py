import streamlit as st
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Load valid credentials from environment variables
    valid_users = {
        os.getenv("ADMIN_USERNAME"): os.getenv("ADMIN_PASSWORD"),
        os.getenv("JAGANNATH_USERNAME"): os.getenv("JAGANNATH_PASSWORD"),
    }

    if st.sidebar.button("Login"):
        if username in valid_users and password == valid_users[username]:
            mlflow.set_experiment("PatentRAGLogin")
            with mlflow.start_run():
                mlflow.log_param("user", username)
            st.session_state["logged_in"] = True
            st.success(f"Welcome {username}")
        else:
            st.error("Invalid username or password")
