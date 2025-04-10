import os
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.2, max_retries=2),
    tools=[DuckDuckGoTools()],
    instructions="Use this agent for market trends, news, and competitor analysis",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama3-70b-8192", api_key=groq_api_key, temperature=0.2, max_retries=2),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use this agent for stock prices, analyst ratings, and financial metrics",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.2, max_retries=2),
    instructions=[
        "Use this routing logic:",
        "1. For questions containing: stock symbols ($), 'financial', 'invest' -> Use Finance Agent",
        "2. For questions about news, trends, or general research -> Use Web Agent",
        "3. For combined queries -> Use both agents sequentially",
        "4. Always show data sources and use tables for financial data"
    ],
    show_tool_calls=True,
    markdown=True,
)

try:
    #agent_team.print_response("What is the current stock price of HP? ", stream=True)
    agent_team.print_response(
    "FINANCE: Compare TSLA and AAPL valuations\n"
    "WEB: Find latest EV market trends\n"
    "COMBINE: Make investment recommendations",
    stream=True
)
except Exception as e:
    print(f"Error: {e}")