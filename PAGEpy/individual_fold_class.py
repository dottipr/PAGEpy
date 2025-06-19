from PAGEpy.multiple_folds_class import KFoldData


class IndividualFold:
    """
    Holds train/test data for a single fold.
    """

    def __init__(self, kfold_data: KFoldData, train_idx, test_idx):
        self.kfold_data = kfold_data
        self.train_idx = train_idx
        self.test_idx = test_idx

        self.x_train = kfold_data.x[train_idx]
        self.x_test = kfold_data.x[test_idx]
        self.y_train = kfold_data.y[train_idx]
        self.y_test = kfold_data.y[test_idx]
        self.selected_features = kfold_data.dataset.selected_features

    def __repr__(self):
        return f"<IndividualFold: {len(self.x_train)} train, {len(self.x_test)} test samples>"
