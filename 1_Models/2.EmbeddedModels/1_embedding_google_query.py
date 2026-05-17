from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', dimensions=32) # create embedding model 

result = embedding.embed_query("Delhi is the capital of India") 

print(result) # output are present in vector formate