from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("5_Data_loader/dl-curriculum.pdf")

docs=loader.load()  # load pdf

spliter= CharacterTextSplitter(
    chunk_size= 200,
    chunk_overlap=0,
    separator=''
)

result = spliter.split_documents(docs)  #use to split documents

print(result[0].page_content)