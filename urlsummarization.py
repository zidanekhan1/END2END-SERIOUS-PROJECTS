import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


st.set_page_config(page_title="URL SUMMARISER BY USING LANGCHAIN")
st.title("Summarize content from youtube or website")
st.subheader("Enter details below")

api_key = st.sidebar.text_input("Enter what URL you wish to be summarized",value="",type="password")

url = st.text_input("Enter whatever URL you desire",label_visibility="collapsed")


llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=api_key)
prompt_template = """
Provide a summary of the following content in 300 words: {text}
"""
prompt = PromptTemplate(template=prompt_template,input_variables=['text'])

if st.button('Summarize content'):
    if not api_key.strip() or not url.strip():
        st.error('Provide the information first')
    elif not validators.url(url):
        st.error("The URL you entered is Invalid!!")
    else:
        try:
            with st.spinner("Suffer while we try to summarize.."):
                if 'youtube.com' in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info=True,language='en',translation='en')
                else:
                    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False,headers=headers)
                docs = loader.load()

                chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output = chain.run(docs)
                st.success(output)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)     