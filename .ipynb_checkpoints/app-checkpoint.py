# app.py
import streamlit as st
from transformers import pipeline
import json

st.set_page_config(page_title="CAT2 AI Assistant", page_icon="🤖")

st.title("AI Question Answering Assistant")

# Load model with caching
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_pipeline()

# Load knowledge base context
@st.cache_data
def load_context():
    try:
        with open("dataset.json", "r") as f:
            data = json.load(f)
            return " ".join([item["content"] for item in data])
    except FileNotFoundError:
        return "Artificial intelligence is a branch of computer science focused on building smart systems."

context = load_context()

# UI Layout
st.subheader("Knowledge Base Context")
with st.expander("View Active Knowledge Base Context"):
    st.write(context)

user_question = st.text_input("Enter your question below:")

if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Processing answer..."):
            result = qa_model(question=user_question, context=context)
            st.success(f"**Answer:** {result['answer']}")
            st.info(f"**Confidence Score:** {result['score']:.4f}")
    else:
        st.warning("Please enter a question first.")