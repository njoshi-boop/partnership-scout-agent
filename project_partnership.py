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

# 1. ENHANCED UI: Page Config & Custom Styling
st.set_page_config(page_title="Partnership Scout AI", page_icon="🤝", layout="wide")

# Custom CSS to make the UI look "SaaS-y"
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .report-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_stats=True)

st.title("🤝 Strategic Partnership Scout")
st.subheader("Autonomous Business Development Agent")
st.write("Leveraging LangGraph and Gemini 2.5-Flash to identify high-leverage brand synergies.")

# Layout for inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Target Company:", "Gymshark", help="The company we are finding partners for.")
    with col2:
        core_audience = st.text_input("Target Demographic:", "Gen Z fitness enthusiasts", help="The core vibe or customer base.")

# 2. UPDATED SYSTEM PROMPT: Removed the Roleplay Intro
system_prompt = """Identify THREE (3) distinct, high-leverage, non-competing partnership opportunities.
                
DO NOT include an introductory sentence like 'As VP of Business Development...' or 'I have identified...'.
Start IMMEDIATELY with '## 1. Target Partner Company: [Name]'.

For EACH recommendation, include:
1. **Target Partner Company Name**
2. **Official Website URL**
3. **The Audience Overlap**
4. **The Deal Concept**
5. **Why It Works**

Use professional Markdown. Keep it data-driven and concise."""

if st.button("Generate Strategy Report"):
    if not company_name:
        st.warning("Please enter a company name.")
    else:
        with st.spinner("Agent searching live web..."):
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
                search = DuckDuckGoSearchRun()
                tools = [search]
                
                agent_executor = create_react_agent(llm, tools)
                
                user_prompt = f"Find 3 different non-competing companies for {company_name} targeting {core_audience}. Provide URLs and synergy reports."
                
                result = agent_executor.invoke({
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                })
                
                # Logic to clean output
                final_response = result["messages"][-1].content
                if isinstance(final_response, list):
                    clean_text = "".join([item.get('text', '') for item in final_response if isinstance(item, dict)])
                else:
                    clean_text = final_response

                # 3. ENHANCED UI: Display in a clean white box
                st.markdown("---")
                st.success("Analysis Complete")
                
                with st.container():
                    st.markdown(f'<div class="report-container">', unsafe_allow_html=True)
                    st.markdown(clean_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 4. BONUS: Download Button
                st.download_button(
                    label="Download Report as TXT",
                    data=clean_text,
                    file_name=f"{company_name}_Partnership_Report.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
