'''
FeatureGenerator.py: Abstract base class for all feature extractors.
Each feature generator must inherit from this class and implement the process() method.
'''

import pandas as pd
import abc  # Provides abstract base class functionality
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureGenerator(abc.ABC):
    def __init__(self, name):
        """
        Initialize feature generator with a name.
        """
        self.name = name

    @abc.abstractmethod
    def process(self, train, test):
        """
        Abstract method: Must be implemented by subclasses.
        This function takes train and test datasets, applies feature extraction,
        and returns the updated train and test DataFrames.
        """
        raise NotImplementedError("Each feature generator must implement the process method.")

    def log(self, message):
        """
        Utility function for logging messages.
        """
        logging.info(f"[{self.name}] {message}")


