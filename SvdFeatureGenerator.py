'''
SvdFeatureGenerator.py: Applies Singular Value Decomposition (SVD) on TF-IDF features
to reduce dimensionality and extract latent semantic features.
'''

from FeatureGenerator import FeatureGenerator
from TfidfFeatureGenerator import TfidfFeatureGenerator
import pandas as pd
import numpy as np
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD

class SvdFeatureGenerator(FeatureGenerator):
    def __init__(self, name='SvdFeatureGenerator', n_components=50):
        """
        Initializes SvdFeatureGenerator with a name and number of components for SVD.
        """
        super().__init__(name)
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, n_iter=15)

    def process(self, train):
        """
        Applies SVD on TF-IDF features from TfidfFeatureGenerator.
        Stores SVD-transformed features directly in train and test DataFrames.
        """
        self.log(f"Applying SVD with {self.n_components} components...")

        # Load TF-IDF features
        tfidfGenerator = TfidfFeatureGenerator('tfidf')
        featuresTrain = tfidfGenerator.read('train')
        xBodyTfidfTrain = featuresTrain[0]

        
        # Stack TF-IDF matrices (no extra split!)
        xBodyTfidf = vstack([xBodyTfidfTrain])

        # Fit and transform SVD
        self.svd.fit(xBodyTfidf)

        self.log("Transforming TF-IDF features with SVD...")
        xBodySvdTrain = self.svd.transform(xBodyTfidfTrain)

        # Convert SVD features to DataFrames
        svd_columns = [f"svd_{i}" for i in range(self.n_components)]
        train_svd_df = pd.DataFrame(xBodySvdTrain, columns=svd_columns, index=train.index)

        # Merge SVD features into train and test DataFrames
        train = pd.concat([train, train_svd_df], axis=1)

        self.log("SVD feature extraction complete.")
        return train
