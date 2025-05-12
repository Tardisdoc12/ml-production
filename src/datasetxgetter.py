################################################################################
# filename: datasetxgetter.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 12/05,2025
################################################################################

import os
import json
from dotenv import load_dotenv
import tweepy
from sqlalchemy import create_engine, Column, Boolean, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

################################################################################

Base = declarative_base()

class Tweet(Base):
    __tablename__ = 'tweets'
    id = Column(String, primary_key=True)
    text = Column(String)
    author_id = Column(String)
    harassment = Column(Boolean)

################################################################################

harassment_keywords_en = [
    "idiot", "dumb", "stupid", "moron", "loser", "fat", "ugly", 
    "kill yourself", "kys", "die", "go to hell", "worthless",
    "freak", "slut", "bitch", "whore", "trash", 
    "retard", "nobody likes you", "no one cares", 
    "shut up", "annoying", "you suck", "hate you", 
    "go away", "loser", "creep", "jerk", "pervert", 
    "harass", "bully", "abuse", "cyberbully"
]

################################################################################

def get_tweet(number_tweets : int = 100, list_words_find : list[str] = harassment_keywords_en) -> list:
    test = load_dotenv()
    if not test:
        raise FileExistsError(".env file is not found or doesn't exist")
    bearer_token = os.getenv("BEARER_TOKEN")
    client = tweepy.Client(bearer_token=bearer_token)
    query = " OR ".join(f'"{kw}"' for kw in list_words_find)
    response = client.search_recent_tweets(
        query=query,
        tweet_fields=["created_at", "author_id", "lang"],
        max_results=number_tweets
    )
    return response

################################################################################

def save_tweet(tweets : dict) -> None:
    engine = create_engine('sqlite:///tweets.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    for t in tweets.data:
        print(t.text)
        a = bool(int(input("est-ce mal (0/1)")))
        tweet = Tweet(
            id= t.id,
            text=t.text,
            author_id=t.author_id,
            harassment = a
        )
        session.merge(tweet)
    session.commit()
    with open("data/raw/data.json", "w", encoding="utf-8") as f:
        json.dump(tweets, f, indent=4, ensure_ascii=False)

################################################################################

def main() -> None:
    tweets = get_tweet(100, harassment_keywords_en)
    save_tweet(tweets)

################################################################################

if "__main__" == __name__:
    main()

################################################################################
# End of File
################################################################################