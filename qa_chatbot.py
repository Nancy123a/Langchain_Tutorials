import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
## streamlit UI
st.set_page_config(page_title="Conservational Q&A Chatbot")
st.header("Hey, Let's chat")

load_dotenv()
chat=ChatOpenAI(openai_api_key=os.environ["openai_key"],
                temperature=0.5)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[SystemMessage(content="You are a comedian AI assistant")]

def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input=st.text_input("input: ",key="input")
response=get_chatmodel_response(input)
submit=st.button("Ask the question")

if submit:
    st.subheader("The Response is")
    st.write(response)