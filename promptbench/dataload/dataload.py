# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .dataset import *

SUPPORTED_DATASETS = [
    "CrimePrediction","Leven"
]

class DatasetLoader:
    
    @staticmethod
    def load_dataset(dataset_name, model, task=None, supported_languages=None):
        """
        Load and return the specified dataset.

        This function acts as a factory method, returning the appropriate dataset object 
        based on the provided dataset name. 
        'math', 'un_multi' and 'iwslt' require additional arguments to specify the languages used in the dataset.

        Args:
            dataset_name (str): The name of the dataset to load.
            task: str: Additional arguments required by 'math'. 
                        Please visit https://huggingface.co/datasets/math_dataset/ to see the supported tasks for math.
            supported_languages: list: Additional arguments required by 'iwslt'. 
                                Please visit https://huggingface.co/datasets/iwslt2017 to see the supported languages for iwslt.
                                e.g. supported_languages=['de-en', 'ar-en'] for German-English and Arabic-English translation.
        Returns:
            Dataset object corresponding to the given dataset_name.
            The dataset object is an instance of a list, each element is a dictionary. Please refer to each dataset's documentation for details.

        Raises:
            NotImplementedError: If the dataset_name does not correspond to any known dataset.
        """
        if dataset_name == "CrimePrediction":
            return CrimePrediction(model)
        elif dataset_name == "Leven":
            return Leven(model)
        else:
            # If the dataset name doesn't match any known datasets, raise an error
            raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")

