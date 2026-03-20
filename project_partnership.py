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
st.subheader("Business Model & Value Proposition Alignment")

# Layout for inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Your Company:", "Gymshark")
    with col2:
        core_audience = st.text_input("Target Demographic:", "Gen Z fitness enthusiasts")

# 1. NEW STRATEGIC PROMPT: Focuses on Business Model and Approach
system_prompt = """Identify THREE (3) distinct strategic partnership opportunities.
                
For EACH recommendation, you must move beyond simple 'audience overlap' and focus on 'Business Model Synergy.'

Output Structure for each partner:
## [Partner Company Name] ([Website URL])

### 1. Value Proposition Alignment
* **The 'Give':** What specific asset, data, or capability does {company_name} provide that this partner is currently missing?
* **The 'Get':** What operational or financial gap does this partner fill for {company_name}? 
* **Business Model Fit:** How does this deal specifically increase LTV (Lifetime Value) or reduce CAC (Customer Acquisition Cost) for both parties?

### 2. The Deal Concept
A detailed description of the product, service, or integration being co-created.

### 3. Strategic Approach (The Pitch)
* **The Hook:** What is the specific 'trigger event' or market trend you would mention in the first outreach email?
* **The Pilot:** Suggest a low-risk, 30-day trial or "Minimum Viable Partnership" to prove value before a full contract.

DO NOT include any introductory sentences. Start immediately with the first partner."""

if st.button("Generate Strategy Report"):
    if not api_key:
        st.error("Please add your GROQ_API_KEY to Streamlit Secrets!")
    elif not company_name:
        st.warning("Please enter a company name.")
    else:
        with st.spinner("Llama 3 Agent is analyzing value propositions and business models..."):
            try:
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
                search = DuckDuckGoSearchRun()
                tools = [search]
                
                agent_executor = create_react_agent(llm, tools)
                
                user_prompt = f"Find 3 non-competing companies for {company_name} targeting {core_audience}. Analyze the business model fit and provide a strategic pitch approach for each."
                
                result = agent_executor.invoke({
                    "messages": [
                        SystemMessage(content=system_prompt.format(company_name=company_name)),
                        HumanMessage(content=user_prompt)
                    ]
                })
                
                clean_text = result["messages"][-1].content

                # 3. UI Display
                st.markdown("---")
                st.success("Analysis Complete")
                st.markdown(clean_text)
                
                # 4. Download Button
                st.download_button(label="Download Report", data=clean_text, file_name=f"{company_name}_Strategic_Report.txt")
                
            except Exception as e:
                st.error(f"Error: {e}")
