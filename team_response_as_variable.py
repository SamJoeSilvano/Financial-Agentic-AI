from typing import Iterator  # noqa
from pydantic import BaseModel
from rich.pretty import pprint
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools.yfinance import YFinanceTools
from agno.utils.pprint import pprint_run_response   

class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str


stock_searcher = Agent(
    name="Stock Searcher",
    model=OpenAIChat("gpt-4o"),
    #response_model=StockAnalysis,
    role="Searches the web for information on a stock.",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
        )
    ],
)


class CompanyAnalysis(BaseModel):
    company_name: str
    analysis: str


company_info_agent = Agent(
    name="Company Info Searcher",
    model=OpenAIChat("gpt-4o"),
    role="Searches the web for information on a stock.",
    #response_model=CompanyAnalysis,
    tools=[
        YFinanceTools(
            stock_price=False,
            company_info=True,
            company_news=True,
        )
    ],
)


team = Team(
    name="Stock Research Team",
    mode="route",
    model=OpenAIChat("gpt-4o"),
    members=[stock_searcher, company_info_agent],
    enable_agentic_context=True,
    markdown=True,
    debug_mode=False,
    show_members_responses=False,
)

#response = team.run("What is the current stock price of NVDA and AAPL?")
#assert isinstance(response.content, StockAnalysis)
#print(response.content)

#response = team.run("What is in the news about NVDA?")
#assert isinstance(response.content, CompanyAnalysis)
#print(response.content)

try:
    #agent_team.print_response("What is the current stock price of HP and Meta? ", stream=True)
    #"WEB: Find latest EV market trends\n"
    #"FINANCE: Compare TSLA and AAPL valuations\n"
    #"COMBINE: Make investment recommendations"

    # Run agent and return the response as a stream
    response_stream: Iterator[RunResponse] = team.run(
    #"WEB: Find latest EV market trends\n"
    "What is the stock price of Nvidia right now"
    #"COMBINE: Make investment recommendations"
    ,stream=True)
    pprint_run_response(response_stream, markdown=True)
except Exception as e:
    print(f"Error: {e}")