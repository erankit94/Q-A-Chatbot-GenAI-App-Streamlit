import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

#["api_keys"]
# st.write("LANGCHAIN_API_KEY:", st.secrets["LANGCHAIN_API_KEY"])
st.write("OPENAI_API_KEY:", st.secrets["OPENAI_API_KEY"])

## LangSmith tracking
# os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Simple Q&A Chatbot with OpenAI"

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant. Please respond to user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    # temperature: 0 to 1: 0 means no creativity
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm, temperature=temperature,
    max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer

## title of App
st.title("Enhanced Q&A chatbot with OpenAI")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password")
api_key=st.secrets["OPENAI_API_KEY"]

# dropdown to select open ai models
llm = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Ask your question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")


#where temp and max token are used in mmodel. why api key is not put in llm?
