"""Wrapper for Reddit PRAW API"""

import time
from typing import Dict

import pandas as pd
import praw


class PostData:
    """
    A class for interacting with the Reddit API
        and extracting post data using PRAW
    """

    def __init__(self, app_creds: Dict):
        self.reddit = praw.Reddit(
            client_id=app_creds["client_id"],
            client_secret=app_creds["client_secret"],
            user_agent=app_creds["user_agent"],
            redirect_uri=app_creds["redirect_uri"],
            client_password=app_creds["password"],
            user_name=app_creds["user_name"],
        )

    def pull_post_data(self, posts) -> pd.DataFrame:
        """
        Extract data from a collection of Reddit posts, including subreddit,
            title, score, number of comments, URL, and all top-level comments,
            and returns it as a pandas DataFrame.

        Parameters
        ----------
        posts: iterable
            Iterable of PRAW submission objects to scrape

        Returns
        -------
        result_df: pd.DataFrame
            DataFrame with the following columns:
            ['subreddit', 'title', 'score', 'num_comments', 'url', 'comments']
        """
        # Pull posts from last year NYCapartments subreddit
        start_time = time.time()
        post_list = []
        for post in posts:
            post_dict = {}
            post_comments = []
            post_dict["subreddit"] = post.subreddit_name_prefixed
            post_dict["title"] = post.title
            post_dict["score"] = post.score
            post_dict["num_comments"] = post.num_comments
            post_dict["url"] = post.shortlink
            post.comments.replace_more(limit=None)
            for top_level_comment in post.comments:
                post_comments.append(top_level_comment.body)
            post_dict["comments"] = post_comments
            post_list.append(post_dict)

        end_time = time.time()
        total_time = round(((end_time - start_time) / 60.0), 2)
        print(f"Finished pulling data in {total_time} mins")

        result_df = pd.DataFrame(post_list)
        return result_df
