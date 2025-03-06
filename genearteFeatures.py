'''
This file serves as the driver for data preprocessing
as well as feature extraction.
'''

import nltk
import os
import pandas as pd
from nltk import ngrams
from nltk.tokenize import word_tokenize
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from sklearn.model_selection import train_test_split

nltk.download('punkt')  # Ensure necessary tokenizer is downloaded

# Function to generate unigrams, bigrams, and trigrams for tweets
def generate_grams(data, print_grams=False):
    """
    Generates unigram, bigram, and trigram features for tweets.
    """
    print("Generating unigrams...")
    data["text_unigram"] = data["text"].map(lambda x: list(word_tokenize(x)))
    
    print("Generating bigrams...")
    data["text_bigram"] = data["text_unigram"].map(lambda x: [' '.join(grams) for grams in ngrams(x, 2)])

    print("Generating trigrams...")
    data["text_trigram"] = data["text_unigram"].map(lambda x: [' '.join(grams) for grams in ngrams(x, 3)])

    # Print samples if needed
    if print_grams:
        print("Sample unigrams:", data["text_unigram"].iloc[0])
        print("Sample bigrams:", data["text_bigram"].iloc[0])
        print("Sample trigrams:", data["text_trigram"].iloc[0])

    return data

# Main function to generate features
def process_all():
    """
    Loads data, generates n-grams, and applies feature extraction methods.
    """
    # Step 1: Load preprocessed data
    print("Loading processed data...")
    train = pd.read_csv("Data/processed_train.csv")
    test = pd.read_csv("Data/processed_test.csv")

    # Step 2: Generate n-gram features
    train = generate_grams(train)
    #test = generate_grams(test)

    # Step 3: Save n-gram enhanced data (for debugging)
    train.to_csv("Data/ngram_train.csv", index=False)
    #test.to_csv("Data/ngram_test.csv", index=False)

    # Step 4: Initialize feature generators
    countFG = CountFeatureGenerator()
    tfidfFG = TfidfFeatureGenerator()
    svdFG = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG = SentimentFeatureGenerator()

    generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]

    # Step 5: Apply feature generators
    for g in generators:
        g.process(train)

    for g in generators:
        g.read('train')
    for g in generators:
        g.read('test')

    # Step 6: Save final feature-enhanced datasets
    train.to_csv("Data/features_train.csv", index=False)
    test.to_csv("Data/features_test.csv", index=False)

    print("Feature generation completed!")

if __name__ == "__main__":
    os.chdir('c:/Users/boyan/OneDrive/Desktop/GIT/NLP_Disaster_Tweets/Twit_Disaster/')  #change environment 
    process_all()
