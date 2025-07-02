"""Script to run sentiment analysis on reddit posts"""

from ast import literal_eval
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from sentiment_functions import interpret_sentiment, predict_sentiment


sns.set_style("whitegrid")


def preprocess_text(text):
    """Preprocess text: lowercase, remove newlines, strip"""
    return text.lower().replace("\n", " ").strip()


DATA_PATH = "./data/reddit_posts_nyc_apt.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data = pd.read_csv(DATA_PATH, index_col=0)

# turn comments to list
data.comments = data.comments.apply(literal_eval)
# Apply preprocessing
data["cleaned_text"] = data["comments"].apply(
    lambda x: " ".join(preprocess_text(comment) for comment in x if comment)
)
start_time = time.time()
# Predict sentiment based on cleaned_text column
# this return sentiment output with logit values
data["sentiment_output"] = data["cleaned_text"].apply(
    lambda text: predict_sentiment(text, model, tokenizer)
)
end_time = time.time()
print(f"Took {end_time-start_time} ms to predict sentiments")
print(f"Took {(end_time-start_time)/60.0} mins to predict sentiments")


# Interpret sentiment output
# get sentiment prediction and probability from output
data["sentiment_dict"] = data["sentiment_output"].apply(interpret_sentiment)
# extract sentiment and probabilities into 2 separate columns
data["sentiment"] = data.sentiment_dict.map(lambda x: x["label"])
data["sentiment_prob"] = data.sentiment_dict.map(lambda x: x["prob"])

# sample some comments and print their sentiment and probabilities
for ix, row in (
    data[["cleaned_text", "sentiment", "sentiment_prob"]].sample(3).iterrows()
):
    print(f"{row['cleaned_text'][:200]}")
    print(f"Sentiment: {row['sentiment']}")
    print(f"Probability: {row['sentiment_prob']}\n")


# Data Viz
# sentiment values
sentiment_df = data.sentiment.value_counts().reset_index()

# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(
    sentiment_df["sentiment"],
    sentiment_df["count"],
    color=["#66BB6A", "#FFCA28", "#EF5350"],
)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution of Reddit Comments")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data.sentiment_prob, ax=ax, kde=True, color="Blue")
_ = ax.set_title("Distribution of Sentiment Probabilities")
_ = ax.set_xlabel("Sentiment Probability")
_ = ax.set_ylabel("Frequency")

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(
    data,
    x="sentiment_prob",
    hue="sentiment",
    ax=ax,
    kde=True,
    hue_order=["negative", "neutral", "positive"],
    palette="Set1",
)
_ = ax.set_title("Distribution of Sentiment Probabilities by Sentiment")
_ = ax.set_xlabel("Sentiment Probability")
_ = ax.set_ylabel("Frequency")


# plot sentiment counts
sentiment_counts = data.sentiment.value_counts().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
g = sentiment_counts.plot(
    kind="barh",
    x="sentiment",
    y="count",
    color=sns.palettes.mpl_palette("Dark2"),
    ax=ax,
)
_ = plt.xlabel("Count")
_ = plt.ylabel("Sentiment")
_ = plt.title("Sentiment Counts")
_ = (
    plt.gca()
    .spines[
        [
            "top",
            "right",
        ]
    ]
    .set_visible(False)
)
