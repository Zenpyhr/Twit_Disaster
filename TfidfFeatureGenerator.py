'''
TfidfFeatureGenerator.py: Extracts TF-IDF features for unigrams, bigrams, and trigrams.
'''

from FeatureGenerator import FeatureGenerator
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureGenerator(FeatureGenerator):
    def __init__(self):
        """
        Initializes TfidfFeatureGenerator with a name.
        """
        super().__init__("TfidfFeatureGenerator")
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)

    def process(self, train, test):
        """
        Extracts TF-IDF features from the text column in train and test datasets.
        Returns the updated train and test DataFrames.
        """
        self.log("Extracting TF-IDF features...")

        # Ensure text column exists
        if "text" not in train.columns or "text" not in test.columns:
            self.log("Error: 'text' column missing in DataFrame!")
            return train, test

        # Fit TF-IDF on train data and transform both train and test
        train_tfidf = self.tfidf_vectorizer.fit_transform(train["text"])
        test_tfidf = self.tfidf_vectorizer.transform(test["text"])

        # Convert sparse matrices to DataFrames
        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())

        # Reset indices to match original data
        train_tfidf_df.index = train.index
        test_tfidf_df.index = test.index

        # Merge TF-IDF features with train and test sets
        train = pd.concat([train, train_tfidf_df], axis=1)
        test = pd.concat([test, test_tfidf_df], axis=1)

        self.log("TF-IDF feature extraction complete.")
        return train, test
