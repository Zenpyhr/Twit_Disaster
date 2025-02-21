import pandas as pd
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")

import re

def clean_text(text):
    if isinstance(text, str):  # Ensure input is a string
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs r"..." makes regex raw strings
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
        return text
    return ""

train["text"] = train["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)

train.fillna("", inplace=True)
test.fillna("", inplace=True)

train.to_csv("Data/processed_train.csv", index=False)
test.to_csv("Data/processed_test.csv", index=False)