from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

templet1= PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
) 

templet2= PromptTemplate(
    template="write 5 topic on {text}",
    input_variables=['text']
) 
parser = StrOutputParser() # convert output into string formate

chain = templet1 | model | parser | templet2 |model | parser # create a chain

result=chain.invoke({'topic':'impact of AI'}) # call the chain
print(result)