# LangChain Learning Repository

This repository is a topic-by-topic LangChain practice workspace built from the code examples present in this folder. The README follows the same sequence as the project structure, so you can study the concepts in order and immediately connect each idea with the matching Python file.

The format used throughout this README is:

- `Topic Name`
- `Theory`
- `Code`

## Project Sequence

```text
LangChain Model/
|-- requirement.txt
|-- 1_Models/
|   |-- 1.ChatModels/
|   `-- 2.EmbeddedModels/
|-- 2_Propmts/
|   |-- Prompt_ui.py
|   `-- Reserach app/
|-- 3_Output_parser/
|-- 4_Chains/
|-- 5_Data_loader/
|-- 6_Text_splitters/
|-- 7_Vector_store/
|-- 8_Retrievers/
`-- 9_RAG/
```

## Setup

Install the dependencies:

```bash
pip install -r requirement.txt
```

Create a `.env` file for the providers used in the scripts:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

Note:

- Most files in this repository use Google Gemini through LangChain.
- `1_chatmodel_openai.py` also needs `langchain-openai` if you want to run the OpenAI example.
- Some examples use Streamlit, Chroma, `pypdf`, `scikit-learn`, and `youtube-transcript-api`.

---

## 1. Models

The `Models` component is the core interface used to talk to AI systems inside LangChain. It standardizes how we work with chat models and embedding models, so we can switch providers without rewriting the full application.

### 1.1 Chat Models

#### Topic Name
OpenAI Chat Model

#### Theory
Chat models are designed for conversational tasks. In LangChain, they receive message-like input and return model responses in a consistent format. This makes it easier to build chat assistants, summarizers, and Q&A systems with different providers.

#### Code
File: `1_Models/1.ChatModels/1_chatmodel_openai.py`

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
result = model.invoke("Write a 5 line poem on cricket")
print(result.content)
```

This example creates an OpenAI chat model and invokes it with a simple prompt.

#### Topic Name
Google Chat Model

#### Theory
LangChain supports provider-specific chat integrations while keeping the usage pattern similar. That means once you understand the chat model interface, moving from OpenAI to Gemini becomes straightforward.

#### Code
File: `1_Models/1.ChatModels/2.chatmodel_google.py`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = model.invoke("what is ai?")
print(result.content)
```

This script shows a basic Gemini chat call using LangChain.

#### Topic Name
Hugging Face Chat Model

#### Theory
Open-source models give more control and flexibility. LangChain can wrap Hugging Face endpoints so they behave like chat models inside the same workflow.

#### Code
File: `1_Models/1.ChatModels/3_chatmodel_huggingface.py`

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)
```

This example uses an open-source model through Hugging Face and wraps it in a chat interface.

### 1.2 Embedding Models

#### Topic Name
Google Embedding for Query

#### Theory
Embeddings convert text into vectors. These vectors capture semantic meaning, which makes them useful for similarity search, retrieval, clustering, and recommendation systems.

#### Code
File: `1_Models/2.EmbeddedModels/1_embedding_google_query.py`

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', dimensions=32)
result = embedding.embed_query("Delhi is the capital of India")
print(result)
```

This script converts a single query into a numeric vector.

#### Topic Name
Google Embedding for Documents

#### Theory
When we embed documents, each document is transformed into a vector. Later, a query vector can be compared with those document vectors to find the most similar content.

#### Code
File: `1_Models/2.EmbeddedModels/2_embedding_google_docs.py`

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', dimensions=32)
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)
print(str(result))
```

This example embeds multiple documents at once.

#### Topic Name
Local Hugging Face Embedding

#### Theory
Local embedding models are useful when you want offline processing, lower latency for small workloads, or more control over deployment. The overall LangChain flow remains the same.

#### Code
File: `1_Models/2.EmbeddedModels/3_embedding_hf_local.py`

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)
print(str(vector))
```

This script uses a local sentence-transformer model for document embeddings.

#### Topic Name
Document Similarity with Cosine Similarity

#### Theory
Once text is converted into embeddings, vector similarity measures such as cosine similarity can be used to identify the closest document to a user query. This is one of the fundamental ideas behind retrieval systems.

#### Code
File: `1_Models/2.EmbeddedModels/4_document_similarity.py`

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
]

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
```

This script embeds cricket-related documents and finds the best match for a query.

---

## 2. Prompts

Prompts are the instructions given to a model to guide its output. LangChain improves prompt engineering by making prompts reusable, structured, and dynamic through prompt templates, chat prompts, and placeholders.

#### Topic Name
Basic Prompt UI

#### Theory
The simplest way to use a prompt is to accept user input and send it directly to a chat model. This is useful for testing prompts quickly and building lightweight interfaces.

#### Code
File: `2_Propmts/Prompt_ui.py`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

st.header("Welcome to prompt web")
user_Input = st.text_input("Enter your prompt here ")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if st.button("Submit"):
    result = model.invoke(user_Input)
    st.write(result.content)
```

This file creates a small Streamlit prompt playground.

#### Topic Name
Prompt Template Based Research App

#### Theory
A `PromptTemplate` lets us define placeholders and fill them at runtime. This is better than hardcoding strings because it improves reuse, readability, validation, and integration with LangChain chains.

#### Code
File: `2_Propmts/Reserach app/Prompt_templet.py`

```python
from langchain_core.prompts import PromptTemplate, load_prompt

template = load_prompt('template.json')

chain = template | model
result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})
```

This app lets the user choose a research paper, explanation style, and summary length, then sends those values into a reusable prompt template.

#### Topic Name
Prompt Generator

#### Theory
Prompt templates can also be saved and reused as external files. This makes prompt management cleaner for larger applications.

#### Code
File: `2_Propmts/Reserach app/prompt_generator.py`

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
""",
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True
)

template.save('template.json')
```

This file creates and saves the reusable prompt used by the research app.

---

## 3. Output Parsers

Structured output and output parsers make model responses much easier to use when they follow a predictable structure such as plain strings, JSON, or validated Python objects.

#### Topic Name
String Output Parser

#### Theory
`StrOutputParser` is the simplest parser. It converts a model response into plain text and is useful when the next step in the chain only needs a clean string.

#### Code
File: `3_Output_parser/01_String_parser.py`

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

templet1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

templet2 = PromptTemplate(
    template="write 5 topic on {text}",
    input_variables=['text']
)

chain = templet1 | model | parser | templet2 | model | parser
```

This script chains two prompts together by passing plain text output from one step into the next.

#### Topic Name
JSON Output Parser

#### Theory
`JsonOutputParser` helps force the model to return structured JSON. This is useful for APIs, automation, dashboards, and any workflow where the output needs to be programmatically consumed.

#### Code
File: `3_Output_parser/02_json_parser.py`

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

templete = PromptTemplate(
    template='Give me 3 important topic on {topic} \n{formate_instruction}',
    input_variables=['topic'],
    partial_variables={'formate_instruction': parser.get_format_instructions()}
)

chain = templete | model | parser
```

This example injects the parser's format instructions into the prompt so the model returns JSON.

#### Topic Name
Pydantic Output Parser

#### Theory
Pydantic adds validation and type safety. `PydanticOutputParser` is useful when the output must match a strict schema before the application uses it.

#### Code
File: `3_Output_parser/03_pydantic_parser.py`

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)
chain = template | model | parser
```

This script validates the model response against a `Person` schema.

---

## 4. Chains

Chains are used when multiple steps must work together as a pipeline. A chain makes complex workflows easier to compose, debug, and reuse.

#### Topic Name
Sequential Chain

#### Theory
A sequential chain runs one step after another. The output of the first step becomes the input of the second step. This is useful for workflows like `generate -> summarize`, `extract -> transform`, or `retrieve -> answer`.

#### Code
File: `4_Chains/01_sequential_chain.py`

```python
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser
```

This file first generates a detailed report and then converts it into a short summary.

#### Topic Name
Parallel Chain

#### Theory
Parallel chains run multiple branches at the same time using the same input. This is helpful when different outputs are needed from the same source text, such as notes, quiz questions, sentiment, or metadata.

#### Code
File: `4_Chains/02_parallel_chain.py`

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain
```

This example generates notes and quiz questions in parallel, then merges them into one final document.

#### Topic Name
Conditional Chain

#### Theory
Conditional chains route the workflow based on logic. This is similar to `if/else` behavior in programming and is useful when the response path depends on classification, sentiment, intent, or document type.

#### Code
File: `4_Chains/03_conditional_chain.py`

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
```

This script first classifies feedback sentiment and then selects the correct response branch.

---

## 5. Data Loaders

Document loaders read data from different sources and convert it into LangChain `Document` objects. These documents then become the input for splitting, embedding, retrieval, and RAG.

#### Topic Name
CSV Loader

#### Theory
`CSVLoader` loads tabular data row by row into document format. It is useful when datasets or structured records need to be searched or summarized.

#### Code
File: `5_Data_loader/01_data_loader.py`

```python
from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(file_path="5_Data_loader/Social_Network_Ads.csv")
csv_docs = csv_loader.load()
print(csv_docs[0].page_content)
```

#### Topic Name
PyPDF Loader

#### Theory
`PyPDFLoader` loads PDF pages into `Document` objects. This is one of the most common starting points for RAG workflows built on notes, reports, or books.

#### Code

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("5_Data_loader/dl-curriculum.pdf")
pdf_docs = pdf_loader.load()
print(pdf_docs[0].page_content)
```

#### Topic Name
Text Loader

#### Theory
`TextLoader` is the simplest loader and is best for plain text files such as notes, logs, transcripts, and articles.

#### Code

```python
from langchain_community.document_loaders import TextLoader

txt_loader = TextLoader(file_path="5_Data_loader/cricket.txt", encoding="utf-8")
txt_docs = txt_loader.load()
print(txt_docs[0].page_content)
```

#### Topic Name
Directory Loader and Lazy Loading

#### Theory
`DirectoryLoader` loads many files from a folder. `lazy_load()` is useful when documents should be processed one by one instead of loading everything into memory at once.

#### Code

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='5_Data_loader/Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

document_docs = loader.lazy_load()
for document in document_docs:
    print(document.metadata)
```

#### Topic Name
Web Base Loader

#### Theory
`WebBaseLoader` extracts visible content from web pages. It is useful when building systems that summarize or analyze online articles or product pages.

#### Code

```python
from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
web_loader = WebBaseLoader(url)
web_docs = web_loader.load()
print(web_docs[0].page_content)
```

This single file demonstrates multiple loader types in one place.

---

## 6. Text Splitters

Large text must be broken into smaller chunks before embedding or generation. Good chunking improves retrieval quality, stays within model limits, and reduces information loss.

#### Topic Name
Length Based Text Splitting

#### Theory
A length-based splitter divides text by chunk size and overlap rules. This is a basic but common strategy for preparing documents for embedding and retrieval.

#### Code
File: `6_Text_splitters/01_length_based.py`

```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("5_Data_loader/dl-curriculum.pdf")
docs = loader.load()

spliter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = spliter.split_documents(docs)
```

#### Topic Name
Text Structure Based Splitting

#### Theory
`RecursiveCharacterTextSplitter` tries to split text more naturally by preserving larger semantic units where possible before falling back to smaller boundaries.

#### Code
File: `6_Text_splitters/02_text_structure_based.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(result)
```

This example splits a short article into manageable chunks.

#### Topic Name
Python Code Splitter

#### Theory
For code, language-aware chunking is better than simple character splitting because it respects functions, classes, and syntax boundaries.

#### Code
File: `6_Text_splitters/03_python_code_split.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(result[1])
```

This script shows how to split Python source while keeping code structure in mind.

#### Topic Name
Semantic Meaning Based Splitting

#### Theory
Semantic chunking groups text by meaning instead of only by size. This can produce more coherent chunks for retrieval, although it is more advanced and often more expensive.

#### Code
File: `6_Text_splitters/04_semantic_meaning_based.py`

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model='text-embedding-004'),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
```

This example chunks text based on semantic shifts in meaning.

---

## 7. Vector Store

A vector store is a system that stores embeddings and retrieves similar vectors efficiently. Vector stores are central to semantic search and RAG systems.

#### Topic Name
Chroma Vector Store

#### Theory
Chroma is a lightweight vector database that works well for local development and small-to-medium applications. In LangChain, it stores document embeddings, supports similarity search, and can keep metadata alongside each document.

#### Code
File: `7_Vector_store/01_langchain_chroma.py`

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001'),
    persist_directory="7_Vector_store/chrome_db",
    collection_name="sample"
)

vector_store.add_documents(docs)
query_result = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
```

This file demonstrates:

- creating `Document` objects
- storing them in Chroma
- running similarity search
- updating a stored document
- deleting a document
- inspecting stored data

---

## 8. Retrievers

Retrievers fetch the most relevant documents for a query. In LangChain, retrievers are runnables, which means they fit naturally into larger chains and RAG pipelines.

#### Topic Name
Vector Store Retriever

#### Theory
A vector store retriever uses embeddings and similarity search to return the most relevant documents from a vector database. This is the most common retriever type used in RAG systems.

#### Code
File: `8_Retrievers/01_vector_store_retriever.py`

```python
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001'),
    collection_name="sample"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("What is LangChain?")
```

This example converts a Chroma vector store into a retriever and returns the top matching documents.

---

## 9. RAG

RAG, or Retrieval-Augmented Generation, is a pattern where a system first retrieves relevant context and then uses that context to generate a grounded answer. This improves factuality, freshness, and usefulness.

#### Topic Name
YouTube Transcript Chatbot

#### Theory
This project combines several LangChain concepts into one full pipeline:

- document acquisition from YouTube transcript data
- text splitting
- embeddings
- vector storage
- retrieval
- prompt design
- answer generation

This is a practical RAG application because the model is instructed to answer only from the retrieved transcript context.

#### Code
File: `9_RAG/01_youtube_chatbot.py`

```python
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

chunks = splitter.create_documents([transcript])
vector_store = Chroma.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = (parallel_chain | prompt | llm | parser)
result = chain.invoke("Can you summarize the video?")
```

This script is the complete end-to-end example in the repository and shows how the earlier topics come together in a real application.

---

## How to Study This Repository

1. Start with `1_Models` to understand how LangChain talks to chat and embedding models.
2. Move to `2_Propmts` to learn how to structure reusable prompts.
3. Study `3_Output_parser` to make model outputs reliable and machine-friendly.
4. Learn `4_Chains` to combine multiple steps into workflows.
5. Use `5_Data_loader` and `6_Text_splitters` to prepare external knowledge.
6. Study `7_Vector_store` and `8_Retrievers` to understand semantic search.
7. Finish with `9_RAG` to see the complete retrieval-to-answer pipeline.

## Final Summary

This repository is a practical LangChain learning path that moves from basic model usage to a complete RAG application. The code in each folder turns the ideas into runnable examples. If you follow the folders in order, you will move from core components to real-world AI workflow design in a clean step-by-step manner.
