import streamlit as st
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

def login():
    # Set background color for sidebar
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                background-color: #b47405;
            }
            
    """, unsafe_allow_html=True)

    # st.sidebar.title("Login")
    st.sidebar.markdown("<h3 style='color: black;'>Login</h3>", unsafe_allow_html=True)
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    # Custom-colored Username and Password labels
    # st.sidebar.markdown("<label class='custom-label'>Username</label>", unsafe_allow_html=True)
    # username = st.sidebar.text_input("", key="username_input")

    # st.sidebar.markdown("<label class='custom-label'>Password</label>", unsafe_allow_html=True)
    # password = st.sidebar.text_input("", type="password", key="password_input")

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
            # st.success(f"Welcome {username}")
            st.markdown(f"<div class='login-success'>✅ Welcome <b>{username}</b></div>", unsafe_allow_html=True)
        else:
            # st.error("Invalid username or password")
            st.markdown("<div class='login-error'>❌ Invalid username or password</div>", unsafe_allow_html=True)


