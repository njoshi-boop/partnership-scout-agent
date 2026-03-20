import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# Securely load API key
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

st.title("🤝 Strategic Partnership Scout AI")
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
        with st.spinner(f"Agent is analyzing {company_name} and searching the web..."):
            try:
                # 1. Initialize Model and Tool
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
                search = DuckDuckGoSearchRun()
                tools = [search]
                
                # 2. Create the modern Prompt Template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an elite VP of Business Development. Your output must be formatted as a 'Partnership Synergy Report' with headings: Target Partner Company, The Audience Overlap, The Deal Concept, Why It Works. Keep it highly actionable and realistic."),
                    ("human", "My company is {company_name}, targeting {core_audience}. Research current market trends for us, identify a highly successful NON-COMPETING company targeting this exact same audience, and develop a concrete concept for a strategic partnership or licensing deal."),
                    ("placeholder", "{agent_scratchpad}"),
                ])

                # 3. Initialize the modern Agent
                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                
                # 4. Run the Agent
                result = agent_executor.invoke({
                    "company_name": company_name,
                    "core_audience": core_audience
                })
                
                st.success("Synergy Report Generated!")
                st.markdown(result["output"])
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
