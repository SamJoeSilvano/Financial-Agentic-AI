from agno.agent import Agent
from agno.models.openai import OpenAIChat
import streamlit as st

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a helpful assistant",
    markdown=True
)

#uery = st.text_input("Ask about market trends:")
#i query:
    #agent.print_response(query)  # Internal processing

# Execute a query
query = "What are the benefits of meditation?"
response = agent.run(query)

# Create a structured run response
#un_response =  agent.create_run_response(response)

# Use the run response
print(response.content)