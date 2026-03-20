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
st.write("An Agentic AI that researches markets and identifies 3 high-leverage business partnerships with live links.")

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
        with st.spinner(f"Agent is investigating {company_name} and searching for 3 partners..."):
            try:
                # 1. Initialize Modern Gemini Model and Search Tool
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
                search = DuckDuckGoSearchRun()
                tools = [search]
                
                # 2. Initialize the LangGraph Agent
                agent_executor = create_react_agent(llm, tools)
                
                # 3. Define the Persona with the 2 NEW requirements:
                #    - Provide THREE partnerships
                #    - Provide the WEBSITE URL
                system_prompt = """You are an elite VP of Business Development. 
                Your goal is to identify THREE (3) distinct, high-leverage, non-competing partnership opportunities.
                
                For EACH of the three recommendations, your output MUST include:
                1. **Target Partner Company Name**
                2. **Official Website URL** (Search the web to find the correct, current link)
                3. **The Audience Overlap** (Explain why the demographics match)
                4. **The Deal Concept** (A concrete, realistic joint-venture or marketing idea)
                5. **Why It Works** (The strategic advantage for both brands)
                
                Use clean Markdown with clear headings. Keep the tone professional, analytical, and ready for a C-suite presentation."""
                
                user_prompt = f"My company is {company_name}, targeting {core_audience}. Find 3 different non-competing companies we should partner with. Provide their URLs and a detailed synergy report for each."
                
                # 4. Execute the Agentic Workflow
                result = agent_executor.invoke({
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                })
                
                st.success("Synergy Report Generated!")
                
                # 5. Clean Output Logic
                final_response = result["messages"][-1].content
                
                if isinstance(final_response, list):
                    # Extract text only if it returns as a list of dicts
                    clean_text = "".join([item.get('text', '') for item in final_response if isinstance(item, dict)])
                    st.markdown(clean_text)
                else:
                    st.markdown(final_response)
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("If you see a rate limit error, wait 30 seconds and try again.")
