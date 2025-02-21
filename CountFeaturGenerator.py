import pandas as pd

def CountFeatureGenerator(train, test):
    def extract_features(df):
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))
        df["char_count"] = df["text"].apply(len)
        df["punctuation_count"] = df["text"].apply(lambda x: sum(1 for char in x if char in "!?,.;"))
        df["hashtag_count"] = df["text"].apply(lambda x: x.count("#"))
        df["mention_count"] = df["text"].apply(lambda x: x.count("@"))
        df["url_count"] = df["text"].apply(lambda x: x.count("http"))
        return df

    train = extract_features(train)
    test = extract_features(test)

    return train, test

