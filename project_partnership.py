import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

# Securely load API key
api_key = st.secrets.get("GEMINI_API_KEY", "YOUR_LOCAL_API_KEY_HERE")
os.environ["GOOGLE_API_KEY"] = api_key

st.title(" Strategic Partnership Scout AI")
st.write("Enter a company. The AI agent will scour the web to find the perfect non-competing partner and draft a joint-venture synergy report.")

col1, col2 = st.columns(2)
with col1:
    company_name = st.text_input("Your Company:", "Gymshark")
with col2:
    core_audience = st.text_input("Core Audience/Vibe:", "Gen Z fitness enthusiasts")

if st.button("Find Partnership Opportunities"):
    if not company_name:
        st.warning("Please enter a company name.")
    else:
        with st.spinner(f"Agent is analyzing {company_name} and searching the web for synergistic partners..."):
            try:
                # Initialize Model and Tool
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
                search = DuckDuckGoSearchRun()
                tools = [search]
                
                # Initialize the ReAct Agent
                agent = initialize_agent(
                    tools, 
                    llm, 
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                    verbose=True,
                    handle_parsing_errors=True
                )
                
                # The Agentic Prompt - This is where the magic happens
                prompt = f"""
                You are a VP of Business Development. Your client is {company_name}, targeting {core_audience}.
                
                Use your search tool to execute the following steps:
                1. Briefly research current market trends for {company_name}.
                2. Identify a highly successful, NON-COMPETING company that targets the exact same '{core_audience}' demographic. 
                3. Develop a concrete concept for a strategic partnership, co-marketing campaign, or licensing deal between the two companies.
                
                Format your output as a 'Partnership Synergy Report' with these exact markdown headings:
                - **Target Partner Company**
                - **The Audience Overlap**
                - **The Deal Concept (How both make money)**
                - **Why It Works (Strategic Moat)**
                
                Keep it highly actionable, business-focused, and realistic. No fluff.
                """
                
                response = agent.run(prompt)
                st.success("Synergy Report Generated!")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"An error occurred: {e}. If this is a rate limit, please wait a minute and try again.")