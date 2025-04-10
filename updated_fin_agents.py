import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini  # Assuming a Gemini class exists in agno.models
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
import time
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import re
from typing import List, Dict, Any # For type hinting

# --- Configuration & Initialization ---

# Load environment variables from a .env file
load_dotenv()

# --- Robust API Key Check ---
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    st.error("🔴 GEMINI_API_KEY not found!")
    st.error("Please ensure you have a .env file in the same directory with GEMINI_API_KEY='your_key'.")
    st.stop() # Stop the application if the key is missing

# --- Cached Resource Initialization ---

@st.cache_resource # Cache the agent team for the session
def initialize_agents() -> Agent:
    """Initializes the multi-agent team."""
    print("--- Initializing Agents (should happen once per session) ---") # For debugging
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for current information, news, and market trends",
        model=Gemini(id="gemini-2.0-flash", api_key=gemini_api_key, temperature=0.2, grounding=True, search=True), # Faster model for web search summaries
        tools=[DuckDuckGoTools()],
        instructions="Focus on retrieving recent and relevant information. Summarize findings concisely in plain text. If you must emphasize, use bold text (`**text**`). Provide source URLs at the end of each summarized point in plain text within parentheses.",
        show_tool_calls=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Retrieve and analyze stock data, financial statements, and key metrics",
        model=Gemini(id="gemini-2.0-flash", api_key=gemini_api_key, temperature=0.1), # Powerful model for financial analysis, low temp
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            stock_fundamentals=True,
            income_statements=True,
            key_financial_ratios=True
        )],
        # *** INSTRUCTION MODIFIED ***
        instructions=[
            "Provide detailed financial information using the available tools.",
            "Present financial data (fundamentals, ratios, income statements) in well-formatted Markdown tables.",
            "Offer brief interpretations or summaries of the data.",
            "IMPORTANT: Clearly state the primary stock ticker symbol (e.g., MSFT, GOOGL) for which you are providing data, usually near the beginning of your response or table.",
            "If multiple tickers are relevant, focus on the main one queried.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

    sentiment_agent = Agent(
        name="Sentiment Analyzer",
        role="Analyze the sentiment of provided text (e.g., news snippets)",
        model=Gemini(id="gemini-2.0-flash", api_key=gemini_api_key, temperature=0.3), # Fast model for sentiment
        instructions="Analyze the sentiment (positive, negative, neutral) of the input text and provide a brief justification.",
        show_tool_calls=False,
        markdown=True,
    )

    # Coordinator Agent
    agent_team = Agent(
        team=[web_agent, finance_agent, sentiment_agent],
        model=Gemini(id="gemini-2.0-flash", api_key=gemini_api_key, temperature=0.2), # Fast model for routing
        instructions=[
            "Carefully analyze the user's query and route it to the most appropriate agent or sequence of agents based on keywords.",
            "Routing Logic:",
            "1. Stock symbols ($, or common patterns like AAPL, MSFT), 'stock', 'financial', 'invest', 'price', 'recommendation', 'earnings', 'fundamentals', 'income statement', 'financial ratios' -> **Finance Agent** is primary. If news is also asked for, use Web Agent first, then Finance Agent.",
            "2. General news, market trends, competitor research -> **Web Agent**.",
            "3. Explicit request to 'analyze sentiment' -> **Sentiment Analyzer** (requires text input, potentially from Web Agent first).",
            "4. Combine results logically if multiple agents are used.",
            "5. Ensure the final response is comprehensive and directly answers the user's query.",
            "6. Prefer Markdown tables for structured financial data presentation (Finance Agent should handle this)."
        ],
        show_tool_calls=True,
        markdown=True,
    )
    return agent_team

@st.cache_resource # Cache the YFinanceTools instance
def get_yfinance_tool() -> YFinanceTools:
    """Creates and caches a YFinanceTools instance."""
    print("--- Initializing YFinance Tool (should happen once per session) ---") # For debugging
    return YFinanceTools(stock_price=True) # Only need stock price capability here

# --- UI and Helper Functions ---

def set_custom_css():
    """Applies custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
        /* [Keep the same CSS as before] */
        .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        .stTextInput input { border-radius: 20px; padding: 10px 15px; }
        .chat-message { padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .user-message { background: #ffffff; border: 1px solid #e0e0e0; }
        .bot-message { background: #007bff; color: white; }
        .stMarkdown table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        .stMarkdown th { background-color: #007bff; color: white; padding: 12px; border: 1px solid #ddd; text-align: left; }
        .stMarkdown td { padding: 12px; border: 1px solid #ddd; text-align: left; }
        .sidebar .stButton button { width: 100%; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

def display_stock_chart(ticker: str, yf_tools: YFinanceTools):
    """Fetches and displays a stock price chart."""
    st.write(f"--- Generating Chart for {ticker} ---") # Info message
    try:
        # Use the cached yfinance tool instance passed as an argument
        stock_data = yf_tools.get_stock_price(tickers=[ticker])

        if stock_data and ticker in stock_data and stock_data[ticker].get('prices'):
            df = pd.DataFrame(stock_data[ticker]['prices'])
            if 'timestamp' not in df.columns or 'close' not in df.columns:
                st.warning(f"Price data for {ticker} is missing expected columns.")
                return
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.dropna(subset=['timestamp', 'close']) # Ensure no NaNs in critical columns
            if not df.empty:
                fig = px.line(df, x='timestamp', y='close',
                             title=f'Recent Stock Price Trend for {ticker}',
                             labels={'timestamp': 'Date', 'close': 'Closing Price'})
                st.plotly_chart(fig, use_container_width=True) # Use container width for responsiveness
            else:
                st.warning(f"No valid price data points found for {ticker} to plot.")
        else:
            st.warning(f"Could not retrieve sufficient stock price data for {ticker} to plot a chart.")
    except Exception as e:
        st.error(f"⚠️ Error displaying stock chart for {ticker}: {e}")

def find_ticker_in_response(text: str) -> str | None:
    """Attempts to find a likely stock ticker symbol in the response text."""
    # Regex to find 1-5 uppercase letters as whole words (common ticker format)
    # Prioritize matches near the start or near markdown tables.
    # This is a heuristic and might need refinement.
    potential_tickers = re.findall(r'\b([A-Z]{1,5})\b', text)

    if not potential_tickers:
        return None

    # Simple heuristic: Often the first ticker mentioned or one near a table is the primary one.
    # Check if common non-tickers were accidentally matched (e.g., 'CEO', 'USA', 'ETF'). Very basic filtering.
    common_non_tickers = {'CEO', 'CFO', 'COO', 'USA', 'ETF', 'LLC', 'INC', 'API', 'NEWS'}
    for ticker in potential_tickers:
        if ticker not in common_non_tickers:
            # Check if the text contains markers of financial data (like a table)
            if '---|---|' in text or 'Recommendation Trend' in text or 'Financial Data' in text:
                print(f"--- Ticker Found in Financial Context: {ticker} ---") # Debugging
                return ticker
            # Fallback: Return first plausible ticker even without table context, might be less reliable
            # print(f"--- Plausible Ticker Found (No Table Context): {ticker} ---") # Debugging
            # return ticker # Optionally enable this less strict fallback

    return None # Return None if no suitable ticker is identified


# --- Main Application Logic ---

def main():
    set_custom_css()

    st.sidebar.title("Financial Intelligence Options")
    if st.sidebar.button("ℹ️ About"):
        st.sidebar.markdown("""
        This AI Assistant uses a team of specialized agents (powered by Gemini & Agno) to answer financial questions, fetch data, and analyze trends.
        - **Web Agent:** Searches the web.
        - **Finance Agent:** Gets stock data & financials (uses yfinance).
        - **Sentiment Agent:** Analyzes text sentiment.
        """)
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Rerun to clear the chat display immediately

    st.title("💰 Financial Intelligence Assistant")
    st.markdown("Ask about stocks (e.g., '$AAPL financials', 'MSFT news and price'), market trends, or sentiment.")

    # Initialize or retrieve chat history from session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Render markdown, allowing tables and other formatting from agents
            st.markdown(message["content"], unsafe_allow_html=True)

    # --- Get Cached Resources ---
    try:
        agent_team = initialize_agents()
        yf_tool_for_charting = get_yfinance_tool()
    except Exception as e:
        st.error(f"🔴 Critical Error during initialization: {e}")
        st.error("Please check API keys and tool configurations.")
        st.stop()

    # Get user input
    if prompt := st.chat_input("Ask your financial question..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_container = st.empty() # Placeholder for streaming effect
            full_response_content = ""
            potential_ticker_for_chart = None

            try:
                # --- Agent Execution ---
                response = agent_team.run(prompt)

                # Extract content (handle potential variations in response object)
                if hasattr(response, 'content'):
                    full_response_content = response.content
                elif isinstance(response, str):
                    full_response_content = response
                else:
                    full_response_content = str(response) # Fallback

                # --- Simulate Streaming Output ---
                display_text = ""
                for i in range(0, len(full_response_content), 5): # Process in chunks
                    chunk = full_response_content[:i+5]
                    display_text = chunk
                    response_container.markdown(display_text + "▌", unsafe_allow_html=True)
                    time.sleep(0.01) # Smaller delay for faster perceived streaming

                # Display final response without cursor
                response_container.markdown(full_response_content, unsafe_allow_html=True)

                # --- Automatic Chart Generation ---
                # Try to find a ticker in the *assistant's* response
                potential_ticker_for_chart = find_ticker_in_response(full_response_content)

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
                full_response_content = f"Sorry, I encountered an error processing your request: {e}"
                response_container.markdown(full_response_content)

            # Add assistant's response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

            # Display chart if a ticker was identified in a relevant response
            if potential_ticker_for_chart:
                # Pass the cached yfinance tool instance
                display_stock_chart(potential_ticker_for_chart, yf_tool_for_charting)


if __name__ == "__main__":
    main()