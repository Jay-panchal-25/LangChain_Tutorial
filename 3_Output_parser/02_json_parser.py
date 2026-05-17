from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = JsonOutputParser()

templete =PromptTemplate(
    template='Give me 3 important topic on {topic} \n{formate_instruction}',
    input_variables=['topic'],
    partial_variables={'formate_instruction':parser.get_format_instructions()}

    )


chain= templete | model|parser

result= chain.invoke({'topic':'AI agent'})

print(result)