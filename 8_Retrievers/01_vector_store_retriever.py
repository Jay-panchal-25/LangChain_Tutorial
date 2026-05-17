from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
vector_store= Chroma.from_documents(
    documents=documents,
    embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001'),# pass embedding model
    collection_name="sample"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
query = "What is LangChain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)