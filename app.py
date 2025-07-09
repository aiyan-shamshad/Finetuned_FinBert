import streamlit as st
from transformers import pipeline
import torch

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model="finetuned_finbert",
        tokenizer="finetuned_finbert",
        device=device
    )

st.title(" FinBERT Sentiment Analysis")
text = st.text_area("Input financial text:", "Apple shares rose 3% after earnings beat estimates.")
classifier = load_model()

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result = classifier(text)[0]
        st.success(
            f"**Sentiment:** {result['label']}\n\n"
            f"**Confidence:** {result['score']:.2%}"
        )