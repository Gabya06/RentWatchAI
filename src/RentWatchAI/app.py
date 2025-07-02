import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sentiment_functions import interpret_sentiment, predict_sentiment, preprocess_text
from RentWatchAI.app.ui_helpers import (
    display_sentiment_result,
    load_model_and_tokenizer,
    sentiment_mapping,
    sentiment_emoji,
)

# processed posts
DATA_PATH = "./data/reddit_posts_nyc_apt_sentiment_roberta.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


# load model
model, tokenizer = load_model_and_tokenizer(model_name=MODEL_NAME)

# Data Processing
data = pd.read_csv(DATA_PATH, index_col=0)

# ---------- APP -----------

st.set_page_config(page_title="Sentiment Analysis Using Roberta Model", layout="wide")
st.title(":star: Sentiment Analysis using Roberta Model")
st.subheader("Are Reddit Comments Positive or Negative?")

st.write("**Data Preview**")

cols = ["title", "cleaned_text"]  # "sentiment", "sentiment_prob"]
if st.button("Show URL", icon="🧠"):
    cols.append("url")

st.dataframe(data=data[cols].head(10), use_container_width=True)


sample_size = st.slider(
    "Number of Posts to Analyze",
    min_value=1,
    max_value=min(20, len(data)),
    value=5,
)
sample_rows = data.head(sample_size).to_dict(orient="records")


if st.button("Analyze Posts and Compare"):
    # avoid division by zero
    max_comments = max(row["num_comments"] for row in sample_rows) or 1

    for i, row in enumerate(sample_rows):
        with st.expander(f"**{row['title']}**", expanded=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                emoji = sentiment_emoji.get(str(row["sentiment"]).lower(), "❓")
                st.markdown(
                    f"<span style='font-size:1.5em'>{emoji}</span><br>"
                    f"<span style='font-size:0.8em'>{row['sentiment'].capitalize()}</span>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown("**Number of Comments**")
                st.progress(
                    int(row["num_comments"]) / max_comments,
                    text=f"{row['num_comments']} comments",
                )

            st.markdown("**Original Post:**")
            st.code(row["cleaned_text"], language="markdown")

            st.markdown("**Sentiment Probability:**")
            st.code(round(row["sentiment_prob"], 2), language="markdown")

user_comment = st.text_area("Enter a comment or post to analyze.", key="text-input")
if st.button("Analyze Comment"):
    if user_comment.strip():
        with st.spinner("Analyzing..."):
            text_input = preprocess_text(user_comment)
            sent_res = predict_sentiment(
                text=text_input, model=model, tokenizer=tokenizer
            )
            sentiment_result_dict = interpret_sentiment(sent_res)
            st.markdown("**Sentiment Analysis Results**")
            display_sentiment_result(sentiment_dict=sentiment_result_dict)
    else:
        st.warning("Please enter a comment or post.")
