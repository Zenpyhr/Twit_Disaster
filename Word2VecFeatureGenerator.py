'''
Word2VecFeatureGenerator.py: Applies Word2Vec embeddings to convert text into numerical features.
'''

from FeatureGenerator import FeatureGenerator
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class Word2VecFeatureGenerator(FeatureGenerator):
    def __init__(self, name='Word2VecFeatureGenerator', vector_size=100, window=5, min_count=2):
        """
        Initializes Word2VecFeatureGenerator with a name and model parameters.
        """
        super().__init__(name)
        self.vector_size = vector_size  # Size of word embedding vectors
        self.window = window  # Context window for Word2Vec
        self.min_count = min_count  # Minimum word occurrence to be included
        self.word2vec_model = None

    def process(self, train):
        """
        Trains a Word2Vec model on the text corpus and generates word embeddings.
        Computes average Word2Vec features for each text instance.
        Stores transformed features directly in train and test DataFrames.
        """
        self.log("Tokenizing text for Word2Vec...")

        # Tokenize text
        train["tokenized_text"] = train["text"].apply(word_tokenize)

        # Combine all text for Word2Vec training
        all_sentences = list(train["tokenized_text"])

        self.log("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(sentences=all_sentences, vector_size=self.vector_size,
                                       window=self.window, min_count=self.min_count, workers=4)

        # Compute average Word2Vec embeddings for each text instance
        train_vectors = train["tokenized_text"].apply(self.get_average_word2vec)

        # Convert to DataFrames
        w2v_columns = [f"w2v_{i}" for i in range(self.vector_size)]
        train_w2v_df = pd.DataFrame(train_vectors.tolist(), columns=w2v_columns, index=train.index)

        # Merge Word2Vec features into train and test DataFrames
        train = pd.concat([train, train_w2v_df], axis=1)

        self.log("Word2Vec feature extraction complete.")
        return train

    def get_average_word2vec(self, words):
        """
        Computes the average Word2Vec embedding for a given list of words.
        """
        valid_words = [word for word in words if word in self.word2vec_model.wv]
        if len(valid_words) == 0:
            return np.zeros(self.vector_size)  # Return zero vector if no words found
        return np.mean([self.word2vec_model.wv[word] for word in valid_words], axis=0)
