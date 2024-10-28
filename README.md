# RentWatchAI

### Overview
RentWatchAI is an NLP-driven project that leverages large language models (LLMs) and Hugging Face transformers like RoBERTa to analyze sentiment in apartment reviews across various subreddits. The goal is to help renters make informed decisions by identifying problematic apartments based on user-generated content. The project uses PRAW to scrape Reddit posts and employs transformers for sentiment scoring and text summarization.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Steps](#steps)
- [Results](#results)
- [Discussion](#discussion)
- [Contributing](#contributing)
- [License](#license)

### Features
- Scrapes Reddit posts using PRAW
- Sentiment analysis with Hugging Face transformers (e.g., RoBERTa)
- Summarizes negative apartment reviews (e.g., OpenAI)
- Displays results in a user-friendly Streamlit app with sentiment scores and charts

## Installation
To get started, clone the repository and install the required libraries:

```bash
git clone https://github.com/yourusername/rentwatchai.git
cd rentwatchai
pip install transformers praw matplotlib pandas streamlit
```

### Usage
Create a Reddit application to get your credentials (client ID, client secret, and user agent).
Update the config.py file with your Reddit credentials.
Run the Streamlit app:
```bash
streamlit run app.py
```

## Steps
### Step 1: Install Required Libraries
Ensure you have the necessary libraries:
```bash
pip install transformers praw matplotlib pandas streamlit
```

### Step 2: Pull Reddit Comments
Use PRAW to pull comments from specific Reddit posts.

### Step 3: Tokenize Comments
Tokenize the comments to prepare them for the RoBERTa model.

### Step 4: Sentiment Analysis
Utilize the RoBERTa model to predict sentiment scores and summarize negative reviews.

### Step 5: Visualize Results
The results are displayed in a user-friendly Streamlit app, providing sentiment scores and visualizations.


## Results
The output includes predicted sentiment scores for each review, a summary of negative reviews, and a bar chart visualizing sentiment distribution across the dataset.

## Discussion
The sentiment analysis reveals insights into user experiences related to various apartments, highlighting both positive and negative sentiments. This information can be valuable for renters and real estate professionals to understand market trends and customer feedback.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.


### Acknowledgements
* [Hugging Face](https://huggingface.co/) for the RoBERTa model.
* [PRAW](https://praw.readthedocs.io/en/latest/) for accessing Reddit data.
* [Matplotlib](https://matplotlib.org/) for data visualization.
* [Streamlit](https://streamlit.io/) for creating the user interface.