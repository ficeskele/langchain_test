from dotenv import load_dotenv
load_dotenv()                                 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

@tool(description="Return a playful weather report for the given city.")

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",  
    temperature=0.3
)

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "i want to konw what is star."}]
})

for msg in result["messages"]:
    print(type(msg).__name__, ":", msg.content)