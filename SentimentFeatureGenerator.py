'''
SentimentFeatureGenerator.py: Extracts sentiment-based features from text.
'''

from FeatureGenerator import FeatureGenerator
import pandas as pd
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure SentimentIntensityAnalyzer is available
nltk.download("vader_lexicon")

class SentimentFeatureGenerator(FeatureGenerator):
    def __init__(self, name="SentimentFeatureGenerator"):
        """
        Initializes SentimentFeatureGenerator with a name.
        """
        super().__init__(name)
        self.sia = SentimentIntensityAnalyzer()

    def process(self, train):
        """
        Extracts sentiment scores from text using VADER and TextBlob.
        Stores transformed features directly in train and test DataFrames.
        """
        self.log("Extracting sentiment features...")

        # Apply sentiment analysis
        train_sentiment = train["text"].apply(self.extract_sentiment_features)
        # Convert to DataFrame

        sentiment_columns = ["sentiment_polarity", "sentiment_subjectivity", "sentiment_vader"]
        train_sentiment_df = pd.DataFrame(train_sentiment.tolist(), columns=sentiment_columns, index=train.index)

        # Merge sentiment features into train and test DataFrames
        train = pd.concat([train, train_sentiment_df], axis=1)

        self.log("Sentiment feature extraction complete.")
        return train

    def extract_sentiment_features(self, text):
        """
        Extracts sentiment polarity, subjectivity, and VADER sentiment score.
        """
        # TextBlob Sentiment Analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # [-1, 1] (negative to positive)
        subjectivity = blob.sentiment.subjectivity  # [0, 1] (factual to opinionated)

        # VADER Sentiment Score
        vader_score = self.sia.polarity_scores(text)["compound"]  # Overall sentiment score

        return [polarity, subjectivity, vader_score]
