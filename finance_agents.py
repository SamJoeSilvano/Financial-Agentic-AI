import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Set up custom CSS
def set_custom_css():
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background: #ffffff;
            border: 1px solid #e0e0e0;
        }
        .bot-message {
            background: #007bff;
            color: white;
        }
        .stMarkdown table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .stMarkdown th {
            background-color: #007bff;
            color: white;
        }
        .stMarkdown td, .stMarkdown th {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize agents
def initialize_agents():
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

    return Agent(
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

# Streamlit app
def main():
    set_custom_css()
    
    st.title("💰 Financial Intelligence Assistant")
    st.markdown("""
    Welcome to your AI-powered financial research assistant! I can:
    - 📈 Analyze stock prices and financial metrics
    - 🔍 Research market trends and news
    - 💡 Provide investment recommendations
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Get user input
    if prompt := st.chat_input("Ask your financial question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Initialize agent team
        agent_team = initialize_agents()
        
        # Generate response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            try:
                # Get the response from the agent
                response = agent_team.run(prompt)
                
                # Extract the content from the response
                if hasattr(response, 'content'):
                    response_content = response.content
                else:
                    response_content = str(response)
                
                # Simulate streaming effect
                for i in range(0, len(response_content), 5):
                    chunk = response_content[:i+5]
                    response_container.markdown(chunk + "▌", unsafe_allow_html=True)
                    time.sleep(0.02)  # Adjust speed here
                
                # Display final response
                response_container.markdown(response_content, unsafe_allow_html=True)
                full_response = response_content

            except Exception as e:
                full_response = f"⚠️ Error: {str(e)}"
                response_container.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()