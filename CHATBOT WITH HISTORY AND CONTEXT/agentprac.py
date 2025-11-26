from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ChatMessageHistory

st.title("GROQ CHATBOT🤖")
st.sidebar.title("API SETTING")
api_key = st.sidebar.text_input("Enter Your API Key")
if not api_key:
    st.info("Please Enter Your API key before beginning")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a friendly chatbot respond to all user queries as good as you can"),
    ("placeholder","{chat_history}"),
    ("user","{question}")
]
)
 
chain = prompt|llm|StrOutputParser()

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        history = ChatMessageHistory()
        for message in st.session_state.messages:
            if message['role']=='user':
                history.add_user_message(message['content'])
            elif message['role'] == 'assistant':
                history.add_ai_message(message['content'])
        store[session_id] = history
    return store[session_id]

with_message_history=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='question',
    history_messages_key='chat_history'
    )

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])



if user_query := st.chat_input("Ask something"):
    st.session_state.messages.append({'role':'user','content':user_query})
    with st.chat_message('user'):
        st.markdown(user_query)
    
    with st.spinner("Thinking..."):
        try:
            response = with_message_history.invoke(
                {"question":user_query},
                config={"configurable":{"session_id":"abc123"}}
            )

            st.session_state.messages.append({"role":"assistant","content":response})
            with st.chat_message("assistant"):
                st.markdown(response)

        except Exception as e:
            st.error(f"An error occured: {e}")
