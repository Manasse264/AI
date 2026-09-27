# app.py
import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from duckduckgo_search import DDGS

st.set_page_config(page_title="AI Question Answering Assistant", page_icon="🤖")

st.title("🤖 AI Question Answering Assistant")
st.write("CAT2 Assessment Practical Project - ITLPA701")

# 1. Load Extractive Model for QA Synthesis
@st.cache_resource
def load_qa_model():
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

tokenizer, qa_model = load_qa_model()

# 2. Silent Web Research Function (No Links Displayed)
def perform_web_research(query, max_results=4):
    research_snippets = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            for res in results:
                snippet = res.get("body", "").strip()
                if snippet:
                    research_snippets.append(snippet)
    except Exception as e:
        st.error(f"Research error encountered: {e}")
    
    # Combine all researched web snippets into a single context text block
    combined_research = " ".join(research_snippets)
    return combined_research

# 3. Extract Answer from Model
def extract_qa_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer

# --- UI Interface ---
mode = st.radio("Select Knowledge Source:", ["Local IT Dataset", "Live Web Search"])

user_question = st.text_input("Enter your question below:")

if st.button("Get Answer"):
    if user_question.strip():
        if mode == "Live Web Search":
            with st.spinner("AI is conducting web research..."):
                # Step 1: Silent web research
                research_context = perform_web_research(user_question)
                
                if not research_context:
                    st.warning("The AI could not find relevant web information for this query.")
                else:
                    # Step 2: Model synthesizes answer directly from researched content
                    answer = extract_qa_answer(user_question, research_context)
                    
                    if not answer.strip():
                        st.warning("The AI researched the web but could not synthesize a clear answer.")
                    else:
                        st.subheader("AI Researched Response:")
                        st.success(f"**Answer:** {answer}")
        else:
            # Local IT Dataset Mode
            with st.spinner("Searching Local Knowledge Base..."):
                try:
                    with open("dataset.json", "r") as f:
                        data = json.load(f)
                        context = " ".join([item["content"] for item in data])
                except FileNotFoundError:
                    context = "Artificial intelligence is a branch of computer science focused on building smart systems."
                
                answer = extract_qa_answer(user_question, context)
                st.subheader("Local Dataset Response:")
                st.success(f"**Answer:** {answer}")
    else:
        st.warning("Please enter a question first.")