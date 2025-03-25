from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

class MultipleFolds:
    """
    A class to generate stratified K-fold splits for training and testing data.

    Attributes:
        rc_data (object): An object containing x_train and y_train as DataFrames.
        folds_count (int): Number of folds for cross-validation.
        x_train_folds (list): Stores the training feature sets for each fold.
        x_test_folds (list): Stores the testing feature sets for each fold.
        y_train_folds (list): Stores the training target sets for each fold.
        y_test_folds (list): Stores the testing target sets for each fold.
        X (DataFrame): The input features from rc_data.
        y (DataFrame): The target variable from rc_data.
        y_stratify (Series): The column used for stratification in cross-validation.
    """

    def __init__(self, input_data, folds_count=5):
        """
        Initializes the rcFolds object and automatically generates stratified K-folds.

        Args:
            rc_data (object): An object containing x_train and y_train as DataFrames.
            folds_count (int, optional): The number of folds for cross-validation (default is 5).
        """
        self.input_data = input_data
        self.folds_count = folds_count

        # Initialize lists to store train-test splits
        self.x_train_folds = []
        self.x_test_folds = []
        self.y_train_folds = []
        self.y_test_folds = []

        # Extract features and target data
        self.X = input_data.x_train  # Assumed to be a DataFrame
        self.y = input_data.y_train  # Assumed to be a DataFrame

        # Generate the folds immediately upon initialization
        self.get_folds()

        # Pass the list of all genes for the purpose of feature selection during ANN model construction
        self.genes_list = self.input_data.genes_list

    def get_folds(self):
        """
        Splits the data into stratified K-folds and stores the train-test sets in lists.
        """
        skf = StratifiedKFold(n_splits=self.folds_count, shuffle=True, random_state=None)
    
        # Loop through each fold's train-test split
        for train_index, test_index in skf.split(self.X, self.y):
            # Check if X is a DataFrame or NumPy array and index accordingly
            if isinstance(self.X, pd.DataFrame):
                self.x_train_folds.append(self.X.iloc[train_index])  # Use iloc for DataFrame
                self.x_test_folds.append(self.X.iloc[test_index])
            else:  # Assume it's a NumPy array
                self.x_train_folds.append(self.X[train_index])  # Use array slicing
                self.x_test_folds.append(self.X[test_index])
    
            # y should always be a DataFrame, so we keep iloc here
            self.y_train_folds.append(self.y[train_index])
            self.y_test_folds.append(self.y[test_index])
