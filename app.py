import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Configuration - Corporate Branding
st.set_page_config(page_title="Generative AI Systems | GA_01", layout="wide")

# Custom CSS for Industrial Aesthetic
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stTextArea textarea { border: 1px solid #e0e0e0; font-family: 'Courier New', monospace; }
    .stButton>button { 
        background-color: #000000; color: white; border-radius: 0px; 
        width: 100%; font-weight: bold; border: none; height: 3em;
    }
    .stButton>button:hover { background-color: #333333; }
    .output-card { 
        background-color: #f9f9f9; padding: 30px; border-left: 10px solid #000; 
        font-size: 1.1rem; line-height: 1.8; color: #333;
    }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_engine():
    # Large model provides the best logic-to-size ratio for portfolios
    model_name = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def generate_technical_sequence(prompt, length, temp, top_p):
    tokenizer, model = initialize_engine()
    
    # SYSTEM ANCHOR: We force the model to start with a factual definition 
    # This prevents the "autocomplete random website" behavior.
    anchor = f"Technical definition of {prompt}: {prompt} is defined as"
    
    inputs = tokenizer.encode(anchor, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=length,
            temperature=temp,
            top_p=top_p,
            do_sample=True,
            # Contrastive Search parameters
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # GUARDRAIL: Cut off if the model drifts into web links or irrelevant metadata
    clean_output = raw_output.replace(anchor, f"{prompt} is defined as")
    if "http" in clean_output:
        clean_output = clean_output.split("http")[0]
    if "\n" in clean_output:
        clean_output = clean_output.split("\n\n")[0]
        
    return clean_output.strip()

# --- SIDEBAR (TECHNICAL CONTROLS) ---
st.sidebar.title("System Parameters")
st.sidebar.markdown("Stochastic configuration for the inference engine.")
token_limit = st.sidebar.slider("Maximum Sequence Length", 50, 300, 150)
temperature = st.sidebar.slider("Sampling Temperature", 0.1, 1.0, 0.5)
top_p_val = st.sidebar.slider("Nucleus Sampling (Top-P)", 0.5, 1.0, 0.9)

# --- MAIN UI ---
st.title("Generative AI System Architecture")
st.text("TRACK: GA | TASK: 01 | INTERN: PRODIGY_INFOTECH")
st.markdown("---")

user_input = st.text_input("Enter Subject Matter", placeholder="e.g. Neural Networks, Quantum Computing")

if st.button("EXECUTE INFERENCE"):
    if user_input:
        with st.status("Initializing engine...", expanded=False) as status:
            result = generate_technical_sequence(user_input, token_limit, temperature, top_p_val)
            status.update(label="Inference complete.", state="complete")
        
        st.markdown("### Generated Output")
        st.markdown(f'<div class="output-card">{result}</div>', unsafe_allow_html=True)
        
        st.download_button("Export Transcript", result, file_name="ga_01_result.txt")
    else:
        st.error("Error: Null input detected. Please enter a subject.")