import json
from RentWatchAI.reddit_api import PostData
from RentWatchAI.config import subreddit_topics


def main():
    """
    Pull data from Reddit API

    To Run:
        python get_data.py
    """
    # open credential file
    with open("client_secrets.json", "r") as f:
        app_creds = json.load(f)

    # connect to Reddit API using PRAW
    post_data = PostData(app_creds=app_creds)

    if post_data is None:
        print("Failed to connect to Reddit API. Exiting.")
        return

    # Pull top 200 posts
    subreddit_instance = post_data.reddit.subreddit(subreddit_topics[1])
    posts = subreddit_instance.top(time_filter="year", limit=200)

    df = post_data.pull_post_data(posts=posts)
    print("Pulled top 200 posts from NYCapartments")
    print(df.head())


if __name__ == "__main__":
    main()
