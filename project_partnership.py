import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# Securely load API key from Streamlit Cloud Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

st.set_page_config(page_title="Partnership Scout AI", page_icon="🤝")

st.title("🤝 Strategic Partnership Scout AI")
st.write("An Agentic AI that researches markets and identifies high-leverage business partnerships.")

# Layout for inputs
col1, col2 = st.columns(2)
with col1:
    company_name = st.text_input("Your Company:", "Gymshark")
with col2:
    core_audience = st.text_input("Core Audience/Vibe:", "Gen Z fitness enthusiasts")

if st.button("Generate Strategy Report"):
    if not company_name:
        st.warning("Please enter a company name.")
    else:
        with st.spinner(f"Agent is investigating {company_name} and searching for partners..."):
            try:
                # 1. Initialize Modern Gemini Model and Search Tool
                # Using 2.5-Flash as the industry standard
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
                search = DuckDuckGoSearchRun()
                tools = [search]
                
                # 2. Initialize the LangGraph Agent
                agent_executor = create_react_agent(llm, tools)
                
                # 3. Define the Strategic Persona and Goal
                system_prompt = """You are an elite VP of Business Development. 
                Your output must be formatted as a 'Partnership Synergy Report' with headings: 
                Target Partner Company, The Audience Overlap, The Deal Concept, Why It Works. 
                Use clean Markdown. Keep it highly actionable and realistic."""
                
                user_prompt = f"My company is {company_name}, targeting {core_audience}. Research current market trends for us, identify a highly successful NON-COMPETING company targeting this exact same audience, and develop a concrete concept for a strategic partnership or licensing deal."
                
                # 4. Execute the Agentic Workflow
                result = agent_executor.invoke({
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                })
                
                st.success("Synergy Report Generated!")
                
                # 5. Clean Output Logic (Handles complex JSON blocks from the API)
                final_response = result["messages"][-1].content
                
                if isinstance(final_response, list):
                    # Extract text only from list of dictionaries
                    clean_text = "".join([item.get('text', '') for item in final_response if isinstance(item, dict)])
                    st.markdown(clean_text)
                else:
                    st.markdown(final_response)
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Check your API key in Streamlit Secrets if this persists.")
