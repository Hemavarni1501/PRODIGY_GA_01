import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Professional Page Config
st.set_page_config(page_title="Generative AI System | GA_01", layout="wide")

# 2. Optimized CSS for Professionalism and Visibility
# Using secondaryBackgroundColor for the box so it adapts to Dark/Light mode automatically
st.markdown("""
    <style>
    /* Professional Sidebar & Buttons */
    .stButton > button { 
        width: 100%; font-weight: 600; height: 3em; 
        border-radius: 4px; border: 1px solid #ccc;
    }
    
    /* The Output Box: Uses Streamlit's native background variable for visibility */
    .output-container { 
        padding: 20px; 
        border-radius: 8px; 
        border: 1px solid rgba(150, 150, 150, 0.2);
        background-color: rgba(150, 150, 150, 0.1); /* Subtle adaptive background */
        margin-bottom: 15px;
    }
    
    /* Ensure title is bold and clean */
    h1 { font-family: 'Inter', sans-serif; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    # Small model for stability on Streamlit Cloud
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def run_generation(topic, length, temp):
    tokenizer, model = load_engine()
    # Technical Anchor for better output
    anchor = f"Technical summary of {topic}: {topic} is defined as"
    inputs = tokenizer.encode(anchor, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_new_tokens=length,
        temperature=temp,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(f"Technical summary of {topic}: ", "").strip()

# --- MAIN INTERFACE ---
st.title("Generative AI System Architecture")
st.caption("TRACK: GA | TASK: 01 | PRODIGY INFOTECH INTERNSHIP")
st.markdown("---")

# Layout: 2 Columns
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("System Parameters")
    subject = st.text_input("Enter Subject", placeholder="e.g. Virtualization")
    token_limit = st.slider("Max Length", 50, 200, 100)
    temp_val = st.slider("Temperature", 0.1, 1.0, 0.7)
    if st.button("RUN INFERENCE"):
        if subject:
            with st.spinner("Processing..."):
                st.session_state.result = run_generation(subject, token_limit, temp_val)
        else:
            st.error("Please enter a subject.")

with col2:
    st.subheader("Neural Output")
    if 'result' in st.session_state:
        # DISPLAY AREA: Plain st.write or st.info is often more visible than custom HTML
        # But here is a styled box that ADAPTS to dark/light mode
        st.info(st.session_state.result)
        
        # Download Action
        st.download_button(
            label="Download Log",
            data=st.session_state.result,
            file_name="ga_01_output.txt",
            mime="text/plain"
        )
    else:
        st.markdown("*Awaiting system input...*")