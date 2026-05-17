from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# YouTube video ID
video_id = "Tuw8hxrFBH8"

# Fetch YouTube Transcript
try:
    # Create API object
    ytt_api = YouTubeTranscriptApi()

    # Fetch transcript
    fetched_transcript = ytt_api.fetch(video_id)

    # Convert transcript into raw data
    transcript_list = fetched_transcript.to_raw_data()

    # Convert transcript list into plain text
    transcript = " ".join(chunk["text"]for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")



# Split Transcript into Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript])


# Create Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")



# Store Embeddings in ChromaDB
vector_store = Chroma.from_documents(chunks,embedding_model)


# Create Retriever
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 4})


# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Prompt Template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.

    Answer ONLY from the provided transcript context.

    If the context is insufficient,
    just say you don't know.

    Context:
    {context}

    Question:
    {question}
    """,
    input_variables=["context", "question"]
)


# Format Retrieved Documents
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# Create Parallel Chain
parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })


# Output Parser
parser = StrOutputParser()

# Main Chain
chain = (parallel_chain | prompt | llm | parser)

# Invoke Chain
result = chain.invoke("Can you summarize the video?")
print(result)