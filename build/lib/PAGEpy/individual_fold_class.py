class IndividualFold:
    """
    A class to prepare the data for training and testing an Artificial Neural Network 
    using the stratified folds from an existing fold object.
    
    Attributes:
        folds_object (object): The object containing the folds (e.g., rcFolds).
        current_fold (int): The index of the current fold to be used for training and testing.
        unique_combinations_array (array): An array of unique combinations from the target variable.
        x_train (DataFrame or ndarray): The multidimensional gene expression array for the training set.
        genes_list (list): A list of genes used for subsetting the data.
        x_test (DataFrame or ndarray): The multidimensional gene expression array for the test set.
        y_train (DataFrame or Series): The dataframe of target variables for the training set.
        y_test_outcome (array): The numerical categories of the outcome for the test set.
    """
    
    def __init__(self, folds_object, current_fold=0):
        """
        Initializes the RcFoldForANN object with the data from the specified fold.
        
        Args:
            folds_object (object): The object containing the folds (e.g., rcFolds).
            current_fold (int, optional): The index of the current fold to be used for training and testing. Default is 0.
        """
        self.folds_object = folds_object  # The folds object containing the training and test sets
        self.current_fold = current_fold  # The current fold index
        
        # Extracting relevant data from the folds object
        self.genes_list = self.folds_object.genes_list  # List of genes for subsetting
        self.x_train = self.folds_object.x_train_folds[current_fold]  # Gene expression data for training
        self.x_test = self.folds_object.x_test_folds[current_fold]  # Gene expression data for testing
        self.y_train = self.folds_object.y_train_folds[current_fold]  # Target values for training set
        self.y_test = self.folds_object.y_test_folds[current_fold]  # Numerical outcomes for testing

    def __repr__(self):
        """
        Return a string representation of the RcFoldForANN object.
        """
        return f"RcFoldForANN(current_fold={self.current_fold}, unique_combinations_array={self.unique_combinations_array.shape})"
