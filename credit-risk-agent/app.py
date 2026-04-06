import streamlit as st
import os
import json
import logging
from dotenv import load_dotenv
from logic import assess_credit_risk, extract_risk_category, extract_approval_status, extract_loan_amount
from datetime import datetime
import asyncio
from omni_agent import run_omni_agent

# ============================================================================
# CONFIGURATION
# ============================================================================
load_dotenv()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Credit Analyst", layout="wide")

# ----- NEW NAVIGATION LOGIC -----
app_mode = st.sidebar.selectbox("Choose App Mode:", ["🏦 AI Credit Risk Assistant", "🤖 Omni-Agent (HN & Pets)"])

if app_mode == "🤖 Omni-Agent (HN & Pets)":
    st.title("🤖 Omni-Agent (HackerNews + Petstore)")
    st.markdown("This agent natively routes requests between a **Local MCP Server** (Hacker News) and an **AWS API Gateway** (Pet Store) using **Groq**.")
    
    # Initialize chat history
    if "omni_messages" not in st.session_state:
        st.session_state.omni_messages = []
        
    for msg in st.session_state.omni_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    prompt = st.chat_input("Try asking: 'What are the top 3 HN stories and do you have any cats?'")
    if prompt:
        st.session_state.omni_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Connecting to tools..."):
                response = asyncio.run(run_omni_agent(prompt))
                st.markdown(response)
        st.session_state.omni_messages.append({"role": "assistant", "content": response})
        
    st.stop() # Prevent the Credit Risk app from rendering on this page
# --------------------------------

st.title("🏦 AI Credit Risk Assistant")
st.markdown(
    "Automated risk assessment using **Customer Profile & Uploaded Financial Documents**, "
    "**Customer History from Snowflake**, and a **Fine-Tuned LLM on Databricks.**"
)

# ============================================================================
# STREAMLIT FORM (Single Submit - No reruns on input changes)
# ============================================================================
with st.form("applicant_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    # ============================================================================
    # COLUMN 1: Applicant Fields + Document Upload
    # ============================================================================
    with col1:
        st.header("📋 Applicant Profile")
        
        # Customer ID
        customer_id = st.text_input(
            "Customer ID",
            value="C001"
        )
        
        # Field 1: Gender
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        
        # Field 2: Age
        age = st.number_input("Age", min_value=18, max_value=75, value=45)
        
        # Field 3: Monthly Income (USD)
        monthly_income = st.number_input(
            "Monthly Income (RS)",
            min_value=0,
            value=50000,
            step=100
        )
        
        # Field 4: Income Stability
        income_stability = st.selectbox(
            "Income Stability",
            options=["Low", "Medium", "High"],
            index=1
        )
        
        # Field 5: Profession
        profession = st.selectbox(
            "Profession",
            options=[
                "Unemployed", "Student", "Skilled Worker", "Professional",
                "Manager", "Entrepreneur", "Retired", "Farmer", "Other", "Not Specified"
            ],
            index=4
        )
        
        # Field 6: Type of Employment
        employment_type = st.selectbox(
            "Type of Employment",
            options=[
                "Unemployed", "Self-Employed", "Casual", "Contractual",
                "Permanent", "Part-time", "Seasonal", "Freelance",
                "Full-time", "Internship", "Apprentice", "Not Specified"
            ],
            index=10
        )
        
        # Field 7: Location
        location = st.selectbox(
            "Location",
            options=["Urban", "Semi-Urban", "Rural"],
            index=1
        )
        
        # Field 8: Loan Amount Request (USD)
        loan_amount = st.number_input(
            "Loan Amount Request (RS)",
            min_value=0,
            value=100000,
            step=10000
        )
        
        # Field 9: Current Loan Expenses (USD)
        current_loan_expenses = st.number_input(
            "Current Loan Expenses (RS)",
            min_value=0,
            value=10000,
            step=5000
        )
        
        # Field 10: House Ownership
        house_ownership = st.selectbox(
            "House Ownership",
            options=["No", "Yes"],
            index=0
        )
        
        # Field 11: Car Ownership
        car_ownership = st.selectbox(
            "Car Ownership",
            options=["No", "Yes"],
            index=0
        )
        
        # Field 12: Dependents
        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=5,
            value=2
        )
        
        # Field 13: Credit Score
        credit_score = st.number_input(
            "Credit Score",
            min_value=500,
            max_value=900,
            value=750,
            step=10
        )
        
        # Field 14: Number of Defaults
        no_of_defaults = st.number_input(
            "Number of Defaults",
            min_value=0,
            max_value=6,
            value=0
        )
        
        # Field 15: Has Active Credit Card
        active_credit_card = st.selectbox(
            "Has Active Credit Card",
            options=["No", "Yes"],
            index=0
        )
        
        # Field 16: Property Location
        property_location = st.selectbox(
            "Property Location",
            options=["Urban", "Semi-Urban", "Rural"],
            index=2
        )
        
        # Field 17: Co-Applicant
        co_applicant = st.selectbox(
            "Co-Applicant",
            options=["No", "Yes"],
            index=1
        )
        
        # Field 18: Property Price (USD)
        property_price = st.number_input(
            "Property Price (RS)",
            min_value=0,
            value=100000,
            step=10000
        )

        st.markdown("---")
        st.subheader("📎 Upload Financial Documents")
        uploaded_files = st.file_uploader(
            "Upload supporting documents (PDFs, images, statements, payslips, etc.)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
    
    # ============================================================================
    # COLUMN 2: Submit Button & Instructions
    # ============================================================================
    with col2:
        st.header("🎯 Assessment Control")
        st.write("✅ Fill in all fields and optionally upload financial documents.")
        st.write("🔍 Then click the **Submit** button below to run the analysis.")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        submitted = st.form_submit_button(
            "🚀 Assess Application",
            use_container_width=True,
            type="primary"
        )
    
        # ============================================================================
        # PROCESS ONLY AFTER SUBMIT
        # ============================================================================
        if submitted:
            with st.spinner("⏳ Analyzing customer profile and documents..."):
                try:
                    # Create profile dict with ALL 17 fields
                    profile_dict = {
                        "customer_id": customer_id,
                        "gender": gender,
                        "age": age,
                        "monthly_income_usd": monthly_income,
                        "income_stability": income_stability,
                        "profession": profession,
                        "employment_type": employment_type,
                        "location": location,
                        "loan_request_usd": loan_amount,
                        "current_loan_expenses_usd": current_loan_expenses,
                        "house_ownership": house_ownership,
                        "car_ownership": car_ownership,
                        "dependents": dependents,
                        "credit_score": credit_score,
                        "defaults": no_of_defaults,
                        "active_credit_card": active_credit_card,
                        "property_location": property_location,
                        "co_applicant": co_applicant,
                        "property_price": property_price,
                    }
                    
                    # Convert to text for current LLM prompt (kept for compatibility)
                    profile_lines = [f"{k}: {v}" for k, v in profile_dict.items()]
                    profile_text = "\n".join(profile_lines)

                    logger.info(f"🔍 Assessing application for customer {customer_id}...")

                    # NOTE: For now we just pass uploaded_files through to logic.py.
                    # Later, logic.py will handle S3 upload + OpenAI parsing + Pinecone indexing.
                    decision_text = assess_credit_risk(
                        customer_id,
                        profile_text,
                        uploaded_files=uploaded_files  # new argument for future use
                    )
                    
                    # Parse decision
                    decision_dict = {
                        "approval_status": extract_approval_status(decision_text),
                        "risk_category": extract_risk_category(decision_text),
                        "loan_sanction_amount": extract_loan_amount(decision_text),
                        "reasoning": decision_text,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Results Grid
                    st.subheader("📊 AI Decision Summary")
                    col_decision = st.columns(3)
                    with col_decision[0]:
                        st.metric("Risk Category", decision_dict["risk_category"])
                    with col_decision[1]:
                        st.metric("Approval Status", decision_dict["approval_status"])
                    with col_decision[2]:
                        st.metric("Sanction Amount", decision_dict["loan_sanction_amount"])
                    
                    # Full reasoning
                    st.divider()
                    with st.expander("📝 Full AI Reasoning (Detailed Analysis)"):
                        st.text_area(
                            "Detailed Analysis:",
                            value=decision_text,
                            height=400,
                            disabled=True
                        )
                    
                    # Show submitted profile
                    st.divider()
                    st.subheader("👤 Applicant Profile Used for Assessment")
                    with st.expander("Profile JSON"):
                        st.json(profile_dict)
                    
                    # Show uploaded documents info
                    if uploaded_files:
                        with st.expander("📎 Uploaded Documents"):
                            for f in uploaded_files:
                                st.write(f"- {f.name} ({f.type}, {f.size} bytes)")
                    
                except Exception as e:
                    st.error(f"❌ Error During Processing: {str(e)}")
                    logger.error(f"Error: {e}", exc_info=True)
