# GA_01: Intelligent Text Generation System

## Project Overview
This repository contains a professional-grade text generation system built during my Generative AI Internship at Prodigy Infotech. The system leverages the **GPT-2 Large** (774M parameter) transformer architecture to generate coherent, technical, and contextually relevant text based on user input.

## Technical Architecture
To ensure industry-ready output and mitigate common Large Language Model (LLM) issues such as hallucination and repetitive looping, the system implements:
* **Contrastive Search:** Utilizes `penalty_alpha` and `top_k` to maintain semantic coherence.
* **Prompt Anchoring:** Uses technical "anchors" to guide the model's autocomplete behavior toward factual definitions.
* **Output Guardrails:** Post-processing logic to filter out hallucinated URLs and metadata.

## Tech Stack
* **Language:** Python 3.10+
* **Model:** Hugging Face GPT-2 Large
* **Frontend:** Streamlit (Minimalist UI)
* **Environment:** GitHub Codespaces & Streamlit Cloud

## Live Deployment
🔗 [View Live Application](https://appigyga01-qcfdbiztgvnxgnxyul38fk.streamlit.app/)

## Setup Instructions
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/PRODIGY_GA_01.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run app.py`