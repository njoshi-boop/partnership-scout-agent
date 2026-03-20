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

# 1. Page Config
st.set_page_config(page_title="Partnership Scout AI", page_icon="🤝", layout="wide")

# Custom CSS for button only
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤝 Strategic Partnership Scout")
st.subheader("Autonomous Business Development Agent")
st.write("Leveraging LangGraph and Gemini 2.5-Flash to identify high-leverage brand synergies.")

# Layout for inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Target Company:", "Gymshark")
    with col2:
        core_audience = st.text_input("Target Demographic:", "Gen Z fitness enthusiasts")

# 2. SYSTEM PROMPT
system_prompt = """Identify THREE (3) distinct, high-leverage, non-competing partnership opportunities.
                
DO NOT include any introductory sentences. Start IMMEDIATELY with '## 1. Target Partner Company: [Name]'.

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

                # 3. Clean UI Display (No white bar!)
                st.markdown("---")
                st.success("Analysis Complete")
                
                # We output the text directly to the page now
                st.markdown(clean_text)
                
                # 4. Download Button
                st.download_button(
                    label="Download Report as TXT",
                    data=clean_text,
                    file_name=f"{company_name}_Partnership_Report.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
