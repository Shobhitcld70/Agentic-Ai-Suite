from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent  
from dotenv import load_dotenv
import requests
import os

os.environ['LANGCHAIN_PROJECT'] = 'agent pipiline'

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather data for a given city"""
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return response.json()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

agent = create_agent(
    llm,
    [search_tool, get_weather_data]
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "get the weather data of birthplace of kalpana chawla"}
    ]
})

print(response)
