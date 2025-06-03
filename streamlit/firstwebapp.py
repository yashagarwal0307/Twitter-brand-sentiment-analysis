import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd  # If df_test is used elsewhere

def get_model():
    tokenizer = BertTokenizer.from_pretrained(
        r"C:\Users\yasha\Downloads\bertsentimentanalysis-transformers-default-v1\bert_resume_classifier"
    )
    model = BertForSequenceClassification.from_pretrained(
        r"C:\Users\yasha\Downloads\bertsentimentanalysis-transformers-default-v1\bert_resume_classifier"
    )
    return tokenizer, model 

tokenizer, model = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

user_input = st.text_area("Enter Text to analyze")
button = st.button("Analyze")

label_map = {
    0: "Toxic",
    1: "Non-Toxic",
    2: "Neutral"
}

if user_input and button:
    # Fix: Properly tokenize and move inputs to device
    test_sample = tokenizer(
        [user_input],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    test_sample = {k: v.to(device) for k, v in test_sample.items()}  # <- INSERTED FROM YOUR CODE

    # Get model prediction
    output = model(**test_sample)  # <- FIXED SYNTAX ERROR

    # Logits and prediction
    st.write("Logits", output.logits)
    y_pred = np.argmax(output.logits.detach().cpu().numpy(), axis=1)
    st.write("Prediction:", label_map[y_pred[0]])