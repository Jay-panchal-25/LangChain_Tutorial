from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() # calling .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # create a model

result=model.invoke("what is ai?") # call the model 

print(result.content) 