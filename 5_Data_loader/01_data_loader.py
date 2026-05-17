from langchain_community.document_loaders import CSVLoader,PyPDFLoader,TextLoader,DirectoryLoader,WebBaseLoader


#*------------------------------------------CSV---------------------------------------------------
csv_loader = CSVLoader(file_path="5_Data_loader/Social_Network_Ads.csv")
csv_docs= csv_loader.load() # load all content together
print(csv_docs[0].page_content)
print("===============================================")


#*----------------------------------PDF------------------------------------------------------------
pdf_loader = PyPDFLoader("5_Data_loader/dl-curriculum.pdf")
pdf_docs= pdf_loader.load()
print(pdf_docs[0].page_content)
print("===============================================")



#*------------------------------------TXT----------------------------------------------------------
txt_loader = TextLoader(file_path="5_Data_loader/cricket.txt" ,encoding="utf-8")
txt_docs= txt_loader.load()
print(txt_docs[0].page_content)
print("===============================================")



#*-------------------------------------FOLDER---------------------------------------------------------
loader = DirectoryLoader(
    path='5_Data_loader/Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
document_docs = loader.lazy_load() # load one by one
for document in document_docs:
    print(document.metadata)
print("===============================================")


#*------------------------------------------WEB BASE---------------------------------------------------
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
web_loader = WebBaseLoader(url)
web_docs = web_loader.load()
print(web_docs[0].page_content)