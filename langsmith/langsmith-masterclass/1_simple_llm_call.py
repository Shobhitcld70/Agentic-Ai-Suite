from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
