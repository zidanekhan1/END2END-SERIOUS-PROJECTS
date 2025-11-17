import streamlit as st
import os
from dotenv import load_dotenv

# LangChain + Tools
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# LOAD ENV
load_dotenv()

# HUGGINGFACE KEY
HF_KEY = os.getenv("HUGGINGFACE_API_KEY")
if HF_KEY:
    os.environ["HUGGINGFACE_API_KEY"] = HF_KEY


# ------------------ STREAMLIT UI --------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG Chatbot with PDF Upload + Chat History")
st.write("Upload PDFs → Ask Questions → Enjoy memory-aware chat responses.")


# ------------------ API KEY INPUT --------------------
api_key = st.text_input("Enter your GROQ API key:", type="password")

if not api_key:
    st.warning("Please enter your GROQ API key to continue.")
    st.stop()

llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")


# ------------------ SESSION HISTORY --------------------
if "stores" not in st.session_state:
    st.session_state["stores"] = {}     # {session_id: ChatMessageHistory()}

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["stores"]:
        st.session_state["stores"][session_id] = ChatMessageHistory()
    return st.session_state["stores"][session_id]


session_id = st.text_input("Session ID:", value="default_session")


# ------------------ PDF UPLOAD --------------------
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Load all PDFs
    raw_docs = []
    for file in uploaded_files:
        temp_path = f"./temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        pdf_loader = PyPDFLoader(temp_path)
        raw_docs.extend(pdf_loader.load())

    # Split docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=250)
    splits = splitter.split_documents(raw_docs)

    if len(splits) == 0:
        st.error("❌ No text detected in the PDF. Try another file.")
        st.stop()

    # Vector store
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()


    # ---------------- PROMPTS ----------------

    # Reformulate contextual questions
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, rewrite the question so it stands alone.\n"
         "Do NOT answer, only rewrite when needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Answer questions
    qa_system_prompt = (
        "You are an intelligent assistant. Use ONLY the retrieved context to answer.\n"
        "If you don’t know, say you don't know.\n"
        "Keep answers under 3 sentences.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


    # Wrap in history handler
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


    # ---------------- USER INPUT --------------------
    user_input = st.text_input("Ask something:")
    if user_input:
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.chat_message("assistant").write(response["answer"])

        st.subheader("Chat History (Debug View)")
        st.write(st.session_state["stores"][session_id].messages)

else:
    st.info("Upload PDF files to activate the chatbot.")