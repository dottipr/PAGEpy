import numpy as np
from sklearn.model_selection import StratifiedKFold

from PAGEpy.dataset_class import GeneExpressionDataset

# TODO: controllare se c'è un modo più pythonico per implementare i K folds


class KFoldData:
    """
    Generates stratified K-fold splits from a GeneExpressionDataset.
    """

    def __init__(
        self, dataset: GeneExpressionDataset, n_folds=5, random_state=1
    ):
        self.dataset = dataset
        self.n_folds = n_folds
        self.random_state = random_state

        self.x = dataset.adata[dataset.train_mask, dataset.selected_features].X
        if not isinstance(self.x, np.ndarray):
            self.x = self.x.toarray()
        self.y = dataset.adata.obs.loc[dataset.train_mask, 'Label'].values

        # Compute split indices
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        self._splits = list(skf.split(self.x, self.y))

    def __len__(self):
        return self.n_folds

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_folds:
            raise IndexError("Fold index out of range")
        train_idx, test_idx = self._splits[idx]
        return IndividualFold(self, train_idx, test_idx)

    def __iter__(self):
        for idx in range(self.n_folds):
            yield self[idx]


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

    def reduce_input_features(self, feature_subset_list: list):
        """
        Subset the input data to include only the selected genes.
        """
        # Find indices of selected genes in the complete gene list
        gene_set_indices = np.where(
            np.isin(self.selected_features, feature_subset_list))[0]

        # Subset training and test data to selected genes only
        x_train = self.x_train[:, gene_set_indices]
        x_test = self.x_test[:, gene_set_indices]

        # Labels remain unchanged
        y_train = self.y_train
        y_test = self.y_test

        return x_train, x_test, y_train, y_test

    def __repr__(self):
        return f"<IndividualFold: {len(self.x_train)} train, {len(self.x_test)} test samples>"
