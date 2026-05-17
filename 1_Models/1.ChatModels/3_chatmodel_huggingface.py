from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", #  model path 
    task="text-generation"  # model task
)

model = ChatHuggingFace(llm=llm) # create a model

result = model.invoke("What is the capital of India")

print(result.content)