from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
## take environment variables from .env
load_dotenv()
import streamlit as st
## Function to load OpenAI model and get response

def get_openai_response(question):
    llm=OpenAI(openai_api_key=os.environ["openai_key"],model_name='gpt-3.5-turbo-instruct',temperature=0.3)
    response=llm(question)
    return response

## initialize streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

input=st.text_input("input: ",key="input")
response=get_openai_response(input)
submit=st.button("Ask the question")



## If ask button is clicked
if submit:
    st.subheader("The Response is")
    st.write(response)