import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd

# Setting up the page configuration with title and icon
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")

# Custom CSS for the entire app
st.markdown(
    """
    <style>
    /* General body and app styling */
    html, body, .stApp {
        background-color: black;  /* Black background */
        color: #FFFFFF;  /* White text color */
        font-family: 'Arial', sans-serif;  /* Cleaner font */
    }
    
    /* Styling for headers, titles, and subtitles */
    .css-10trblm, .css-1629p8f, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;  /* White titles */
    }
    
    /* Sidebar background and text styling */
    .css-1d391kg, .css-1d391kg > div {
        background-color: #333333 !important;  /* Dark sidebar */
        color: #FFFFFF !important;  /* White sidebar text */
        border-right: 1px solid #FFFFFF !important;  /* Border for sidebar */
    }

    /* Sidebar input fields, text input, and password fields */
    input, .stTextInput input, .stSidebar button {
        background-color: #444444 !important;  /* Dark input background */
        color: #FFFFFF !important;  /* White input text */
        border: 1px solid #FFFFFF !important;  /* White border */
    }

    /* Placeholder styling */
    ::placeholder {
        color: #CCCCCC !important;  /* Light grey placeholders */
    }

    /* Buttons styling */
    .stButton button {
        background-color: #000000 !important;  /* Black button */
        color: #FFFFFF !important;  /* White text on buttons */
        border-radius: 8px !important;  /* Rounded corners */
        font-weight: bold;
    }
    
    /* Chat message bubbles styling */
    .stChatMessage {
        background-color: #555555 !important;  /* Dark bubble background */
        color: #FFFFFF !important;  /* White text in messages */
        border-radius: 10px !important;  /* Smooth rounded edges */
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Scrollable chat history styling */
    .stChatMessageContainer {
        overflow-y: auto !important;  /* Scroll when needed */
        max-height: 400px;
    }

    /* Styling for info and error messages */
    .stAlert {
        background-color: #FF5722 !important;  /* Red-orange alert background */
        color: #FFFFFF !important;  /* White text */
        font-weight: bold;
    }

    /* Padding and spacing adjustments for content */
    .block-container {
        padding-top: 20px !important;
        padding-left: 30px !important;
        padding-right: 30px !important;
        background-color: #333333 !important;  /* Dark background */
    }

    /* Table styling for SQL results */
    table {
        width: 100%;
        color: #FFFFFF !important;  /* White text for tables */
        border-collapse: collapse;
    }
    
    table, th, td {
        border: 1px solid #FFFFFF !important;  /* White table borders */
        padding: 8px;
    }

    th {
        background-color: #444444;
        font-weight: bold;
    }

    td {
        background-color: #333333;
    }

    /* Intermediate steps and results */
    .stMarkdown {
        color: #FF0000 !important;  /* Red color for intermediate steps */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Setting up the title of the app
st.title("🦜 LangChain: Chat with SQL DB")

# Database connection options
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["Use SQLLite 3 Database- Student.db", "Connect to your MySQL Database"]

# Sidebar options for database selection
selected_opt = st.sidebar.radio(label="Choose the DB you want to chat with", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

# API key input for Groq
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

# Info messages if DB URI or API key is not provided
if not db_uri:
    st.info("Please enter the database information and URI")

if not api_key:
    st.info("Please add the Groq API key")

# LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Database configuration function
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))   

# Configure DB based on user selection
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Creating an agent with SQL DB and Groq LLM
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Session state for messages (clear button available)
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input for user query
user_query = st.chat_input(placeholder="Ask anything from the database")

# If user query is submitted
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Generate response from agent
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Ensure the response is in tabular format
            if isinstance(response, list):
                if all(isinstance(i, tuple) for i in response) and len(response) > 0:
                    # Assuming the first tuple contains the headers
                    headers = [f"Column {i+1}" for i in range(len(response[0]))]
                    df = pd.DataFrame(response, columns=headers)
                    st.dataframe(df.style.set_properties(**{'color': 'white', 'background-color': 'black'}))
                else:
                    st.write("The response is not in tabular format.")
            else:
                st.write("The response is not in tabular format.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
