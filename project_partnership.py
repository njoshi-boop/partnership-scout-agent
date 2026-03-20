import streamlit as st
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# Securely load API key from Streamlit Cloud Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

st.set_page_config(page_title="Partnership Scout AI", page_icon="🤝", layout="wide")

# Custom CSS for the button
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #f63366;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤝 Strategic Partnership Scout")
st.subheader("Powered by Llama 3.3 (High-Speed Agent)")

# Layout for inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Target Company:", "Gymshark")
    with col2:
        core_audience = st.text_input("Target Demographic:", "Gen Z fitness enthusiasts")

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
    if not api_key:
        st.error("Please add your GROQ_API_KEY to Streamlit Secrets!")
    elif not company_name:
        st.warning("Please enter a company name.")
    else:
        with st.spinner("Llama 3 Agent is researching the web at warp speed..."):
            try:
                # 1. Swapped to Groq (Model: llama-3.3-70b-versatile)
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
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
                
                clean_text = result["messages"][-1].content

                # 3. UI Display
                st.markdown("---")
                st.success("Analysis Complete")
                st.markdown(clean_text)
                
                # 4. Download Button
                st.download_button(label="Download Report", data=clean_text, file_name=f"{company_name}_Report.txt")
                
            except Exception as e:
                st.error(f"Error: {e}")
