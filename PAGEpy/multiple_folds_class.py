import numpy as np
from sklearn.model_selection import StratifiedKFold

from PAGEpy.format_data_class import GeneExpressionDataset
from PAGEpy.individual_fold_class import IndividualFold


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

    #     self.folds = []
    #     self._generate_folds()

    # def _generate_folds(self):
    #     """
    #     Splits the data into stratified K-folds.
    #     """
    #     skf = StratifiedKFold(n_splits=self.n_folds,
    #                           shuffle=True, random_state=self.random_state)
    #     for train_idx, test_idx in skf.split(self.x, self.y):
    #         fold = IndividualFold(self, train_idx, test_idx)
    #         self.folds.append(fold)
