import streamlit as st
from typing import Iterator
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.utils.pprint import pprint_run_response   
from agno.team.team import Team
import os
import time
import re
from io import StringIO
import sys
from dotenv import load_dotenv

# Load environment variables
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

agent_team = Team(
    name="Main Agent",
    mode="route",
    members=[web_agent, finance_agent],
    model=Groq(id="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.2, max_retries=2),
    instructions=[
        "Use this routing logic:",
        "1. For questions containing: stock symbols ($), 'financial', 'invest' -> Use Finance Agent",
        "2. For questions about news, trends, or general research -> Use Web Agent",
        "3. For combined queries -> Use both agents sequentially",
        "4. Always show data sources and use tables for financial data"
    ],
    enable_agentic_context=True,
    debug_mode=False,
    markdown=True,
)

try:
    #agent_team.print_response("What is the current stock price of HP and Meta? ", stream=True)
    #"WEB: Find latest EV market trends\n"
    #"FINANCE: Compare TSLA and AAPL valuations\n"
    #"COMBINE: Make investment recommendations"

    # Run agent and return the response as a stream
    response_stream: Iterator[RunResponse] = agent_team.run(
    #"WEB: Find latest EV market trends\n"
    "What is the stock price of Nvidia right now"
    #"COMBINE: Make investment recommendations"
    ,stream=True)
    pprint_run_response(response_stream, markdown=True)
except Exception as e:
    print(f"Error: {e}")