## integrate our code ith openai API
import os
from constants import openai_key
from langchain_openai import OpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory ## to make llm model rememeber the conservation

os.environ['OPENAI_API_KEY']=openai_key

st.title('Celebrity Search Results')
input_text=st.text_input('Search the topic u want')

llm=OpenAI(temperature=0.3)

## prompt template
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

second_input_prompt=PromptTemplate(
    input_variables=['person_info'],
    template="From the following text, extract only the full name of the persona and their date of birth:{person_info}"
)

## memory
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')


chain1 = first_input_prompt | llm

# LCEL Chain 2: Extract DOB from the previous output
chain2 = second_input_prompt | llm

#parent_chain = chain1 | chain2
#parent_chain = RunnableSequence(chain1, chain2)

chain1_llmchain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person_info',memory=person_memory)
chain2_llmchain = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

# Parent chain (still supported, but now using the correct LLMChain instances)
## get the entire form in json form
parent_chain = SequentialChain(chains=[chain1_llmchain, chain2_llmchain],input_variables=['name'],output_variables=['person_info','dob'], verbose=True)


if input_text:
    st.write(parent_chain.invoke({'name': input_text}))
    st.subheader("Conversation History:")
    st.write(person_memory.buffer)