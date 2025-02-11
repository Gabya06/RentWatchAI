from ast import literal_eval
import time
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
import seaborn as sns

import torch
import torch.nn.functional as F
import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


def predict_sentiment(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> transformers.modeling_outputs.SequenceClassifierOutput:
    """
        Predicts the sentiment of a text using a pre-trained model and tokenizer.

    Parameters
    ----------
    text: str
        String with input text
    model: transformers.AutoModelForSequenceClassification
        The pre-trained model.
    tokenizer: transformers.AutoTokenizer
        The pre-trained tokenizer.
    max_length: (int, optional)
        The maximum sequence length accepted by the model. Default is 512.

    Returns
    -------
    transformers.SequenceClassifierOutput
      The model's output, including predicted labels and probabilities.
    """

    # Tokenize the text
    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # Check if any of the token IDs in the input are the unknown token ID (OOV).
    input_ids = encoded_input["input_ids"]
    oov_indices = [
        i
        for i, token_id in enumerate(input_ids[0])
        if token_id == tokenizer.unk_token_id
    ]
    if oov_indices:
        print(
            f"Warning: Found {len(oov_indices)} OOV tokens at positions: {oov_indices} in text: {text}"
        )

    # Get position IDs and check for out-of-range positions
    # where the positions are greater than vocab size
    position_ids = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    )
    out_of_range_positions = [
        i for i, pos_id in enumerate(position_ids) if pos_id >= tokenizer.vocab_size
    ]
    if out_of_range_positions:
        print(
            f"Warning: Found {len(out_of_range_positions)} position IDs out of range: {out_of_range_positions} in text: {text}"
        )

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation during inference
        output = model(**encoded_input)

    return output


def interpret_sentiment(output):
    """
    Interprets the sentiment from the model's output.

    Parameters
    ----------
        output: transformers.SequenceClassifierOutput:
        The model's output.

    Returns
    -------
        dict: A dictionary containing the predicted sentiment label and its probability.
               e.g., {'label': 'positive', 'prob': 0.95}
    """
    # Calculate the probabilities of each sentiment using softmax
    probs = F.softmax(output.logits, dim=1)

    # Get the predicted label (index of the highest probability)
    predicted_label_index = torch.argmax(probs).item()

    # Map the label index to a sentiment label
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_label = sentiment_mapping.get(predicted_label_index, "unknown")

    # Get the probability of the predicted label
    predicted_prob = probs[0][predicted_label_index].item()

    # Return the label and probability as a dictionary
    return {"label": predicted_label, "prob": predicted_prob}
