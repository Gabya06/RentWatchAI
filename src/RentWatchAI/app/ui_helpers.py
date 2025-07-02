from typing import Dict
import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Sentiment to emoji mapping
sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}

# Add colored bar based on sentiment
sentiment_mapping = {
    "Positive": "#4CAF50",  # Green
    "Negative": "#F44336",  # Red
    "Neutral": "#FFEB3B",  # Yellow
}


@st.cache_resource
def load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer

    Parameters:
    ----------
    model_name: str
        Hugging Face model name, i.e:

    Returns:
    --------
    model:
    tokenizer:
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def display_sentiment_result(sentiment_dict):
    """
    Helper function to display sentiment results
    """
    emoji = sentiment_emoji.get(str(sentiment_dict["label"]).lower(), "❓")
    sentiment = sentiment_dict["label"].capitalize()
    prob = round(sentiment_dict["prob"], 2)
    sentiment_color = sentiment_mapping.get(sentiment, "#BDBDBD")
    st.markdown(
        f"<span style='font-size:1.5em'>{emoji}</span><br><b>{sentiment}</b>\
              ({prob})",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='height: 16px; width: 25%;\
            background-color: {sentiment_color}; border-radius: 6px; \
                margin-top: 8px;'></div>",
        unsafe_allow_html=True,
    )
