'''
CountFeatureGenerator.py: Extracts basic count-based features from text.
Includes word count, character count, punctuation count, hashtag count, mention count, URL count,
as well as unigram, bigram, and trigram counts.
'''

from FeatureGenerator import FeatureGenerator
import pandas as pd
import string

class CountFeatureGenerator(FeatureGenerator):
    def __init__(self):
        """
        Initializes CountFeatureGenerator with a name.
        """
        super().__init__("CountFeatureGenerator")

    def process(self, train, test):
        """
        Extracts count-based features from the text column in train and test datasets.
        Returns the updated train and test DataFrames.
        """
        self.log("Starting feature extraction...")

        # Apply feature extraction functions to both train and test datasets
        train = self.extract_count_features(train)
        test = self.extract_count_features(test)

        self.log("Feature extraction complete.")
        return train, test

    def extract_count_features(self, df):
        """
        Computes various count-based features from the text column in the given DataFrame.
        """
        # Ensure the text column exists
        if "text" not in df.columns:
            self.log("Error: 'text' column missing in DataFrame!")
            return df

        self.log("Computing word count...")
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

        self.log("Computing character count...")
        df["char_count"] = df["text"].apply(lambda x: len(str(x)))

        self.log("Computing punctuation count...")
        df["punctuation_count"] = df["text"].apply(lambda x: sum(1 for char in str(x) if char in string.punctuation))

        self.log("Computing hashtag count...")
        df["hashtag_count"] = df["text"].apply(lambda x: str(x).count("#"))

        self.log("Computing mention count...")
        df["mention_count"] = df["text"].apply(lambda x: str(x).count("@"))

        self.log("Computing URL count...")
        df["url_count"] = df["text"].apply(lambda x: str(x).count("http"))

        # N-gram features
        self.log("Computing unigram count...")
        df["unigram_count"] = df["text_unigram"].apply(len)

        self.log("Computing bigram count...")
        df["bigram_count"] = df["text_bigram"].apply(len)

        self.log("Computing trigram count...")
        df["trigram_count"] = df["text_trigram"].apply(len)

        return df
