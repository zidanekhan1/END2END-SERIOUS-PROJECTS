import streamlit as st
from pathlib import Path
from langchain_classic.agents import initialize_agent,create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3 
from langchain_groq import ChatGroq

st.title("LANGCHAIN🔗: CHAT WITH SQLDB" )
LOCAL_DB="USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLlite3 Database - student.db","Use MYSQL Database"]


selected_opt = st.sidebar.radio(label="SELECT THE DATABASE YOU WANT",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("enter your mysql host")
    mysql_user=st.sidebar.text_input("enter your MYSQL username")
    mysql_password=st.sidebar.text_input("enter your mysql password",type="password")
    mysql_db=st.sidebar.text_input("enter your database name")
else:
    db_uri=LOCAL_DB

api_key = st.sidebar.text_input("enter your groq api key",type="password")

if not db_uri:
    st.info("provide a database big bro")

if not api_key:
    st.warning("where is your api big bro")
    st.stop()

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LOCAL_DB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("please provide all MySQL connection details")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)

llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key,streaming=True)

tooklit = SQLDatabaseToolkit(db=db,llm=llm)

agent = create_sql_agent(llm=llm,toolkit=tooklit,
                         verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"assistant","content":"How may i help you"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.text_input("ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)