from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
st.header("Welcome to prompt web")
user_Input=st.text_input("Enter your prompt here ")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if st.button("Submit"):
    result=model.invoke(user_Input)
    st.write(result.content)
