'''format the data for the neural network'''

import fnmatch
import os
import pickle

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class GeneExpressionDataset:
    """
    A class to handle gene expression datasets, including loading, preprocessing,
    feature selection, and train-test splitting.
    """

    def __init__(
        self,
        data_dir: str = '/your/local/dir/data_folder/',
        counts_pattern: str = "*counts.mtx",
        barcodes_pattern: str = "*barcodes.txt",
        genes_pattern: str = "*genes.txt",
        metadata_pattern: str = "*infection_status.csv",
        test_set_size=0.2,
        random_seed=1,
        hvg_count=1000,
        pval_cutoff=0.01,
        gene_selection='HVG',
        pval_correction='bonferroni',
        features_out_filename: str = "feature_set.pkl",
        train_samples_out_filename: str = "train_samples.txt",
    ):
        """
        Initializes the GeneExpressionDataset class with specified parameters.

        Parameters:
        - data_dir (str): Path to the holder containing the neccesary files.
        - test_set_size (float): Fraction of data to be used as a test set (default: 0.2).
        - random_seed (int): Seed for reproducible dataset splits (default: 1).
        - hvg_count (int): number of HVGs for selection
        - gene_selection (str): method of feature selection can either be 'HVG' or 'Diff'
        """

        # Directory and file patterns
        self.data_dir = data_dir
        self.counts_path = self._find_file(counts_pattern)
        self.barcodes_path = self._find_file(barcodes_pattern)
        self.genes_path = self._find_file(genes_pattern)
        # labels_path = find_file("*target_variable.csv")[0] # MODIFIED !!
        self.metadata_path = self._find_file(metadata_pattern)
        self.features_out_fn = features_out_filename
        self.train_samples_out_fn = train_samples_out_filename

        # Dataset parameters
        self.test_set_size = test_set_size
        self.random_seed = random_seed
        self.hvg_count = hvg_count
        self.pval_cutoff = pval_cutoff
        self.gene_selection_method = gene_selection
        self.pval_correction = pval_correction

        # Load and prepare AnnData
        self.adata = self._load_and_normalize_data()
        self.genes_list = self.adata.var_names.to_list()

        # Encode labels and prepare splits
        self._encode_labels()
        self.train_mask, self.test_mask = self._split_train_test()

        # Select features and scale data
        self.selected_features = self._select_features()

        # Initialize attributes for train-test splits and data
        self.scaler = None
        self.x_train, self.x_test, self.y_train, self.y_test = self._prepare_scaled_data()

    def _find_file(self, pattern: str) -> str:
        files = os.listdir(self.data_dir)
        matches = fnmatch.filter(files, pattern)
        if not matches:
            raise FileNotFoundError(
                f"No file matching {pattern} in {self.data_dir}")
        return os.path.join(self.data_dir, matches[0])

    def _load_and_normalize_data(self) -> sc.AnnData:
        """
        Loads and prepares data from matrix, barcodes, genes, and metadata files.

        Args:
            counts_pattern (str): Glob pattern for the matrix file.
            barcodes_pattern (str): Glob pattern for the barcodes file.
            genes_pattern (str): Glob pattern for the genes file.
            metadata_pattern (str): Glob pattern for the metadata file.

        Returns:
            AnnData: The constructed AnnData object (also assigned to self.adata).

        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If data shapes are inconsistent or loading fails.
        """
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        try:
            adata = sc.AnnData(
                X=mmread(self.counts_path).tocsc().T,
                obs=pd.read_csv(self.metadata_path),  # dataframe
            )

            # observations are barcodes (data points):
            adata.obs_names = pd.read_csv(
                self.barcodes_path, header=None, sep="\t")[0].values

            # Remove 'Sample' column from obs, as it's duplicated
            if 'Sample' in adata.obs.columns:
                adata.obs.drop(columns=['Sample'], inplace=True)

            # variables are gene names (data features):
            adata.var_names = pd.read_csv(
                self.genes_path, header=None, sep="\t")[0].values

            print("AnnData object successfully constructed.")

            # Normalize each cell by total counts (to 10,000 counts per cell)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            print('AnnData object counts are now normalized.')

        except Exception as e:
            raise ValueError(f"Failed to construct anndata object: {e}") from e

        return adata

    def _encode_labels(self):
        """
        Encodes the target variable into numerical values.
        """
        label_encoder = LabelEncoder()
        self.adata.obs['Label'] = label_encoder.fit_transform(
            self.adata.obs['Status'])

    def _split_train_test(self) -> tuple:
        """
        Splits the dataset using AnnData indexing and stores split information
        in .obs
        """
        train_indices, test_indices = train_test_split(
            np.arange(self.adata.n_obs),
            test_size=self.test_set_size,
            random_state=self.random_seed,
            stratify=self.adata.obs['Label']
        )

        # Store split information in AnnData.obs
        self.adata.obs['split'] = 'test'  # Default to test
        self.adata.obs.loc[self.adata.obs_names[train_indices],
                           'split'] = 'train'

        print(
            f"Training samples: {len(train_indices)}, "
            f"Test samples: {len(test_indices)}")

        # Save training sample names
        with open(self.train_samples_out_fn, 'w', encoding='utf-8') as f:
            training_samples = pd.Series(
                self.adata.obs_names[train_indices])
            for name in training_samples.tolist():
                f.write(f"{name}\n")

        train_mask = self.adata.obs['split'] == 'train'
        test_mask = self.adata.obs['split'] == 'test'

        return train_mask, test_mask

    def _select_features(self):
        """
        Selects features (genes) from training data using HVG or differential
        expression.
        """
        # Selects features from only the training set (to avoid data leakage)
        adata_train = self.adata[self.adata.obs['split'] == 'train'].copy()

        if self.gene_selection_method == 'HVG':
            # Compute HVGs
            sc.pp.highly_variable_genes(
                adata_train, n_top_genes=self.hvg_count, n_bins=100)
            # Get the list of HVGs
            selected_features = adata_train.var_names[
                adata_train.var['highly_variable']].tolist()

        elif self.gene_selection_method == 'Diff':
            sc.tl.rank_genes_groups(
                adata_train, groupby='Status', method='t-test',
                key_added="t-test", corr_method=self.pval_correction)
            sig_genes = sc.get.rank_genes_groups_df(
                adata_train, group=self.adata.obs['Status'][0], key='t-test',
                pval_cutoff=self.pval_cutoff)['names']
            selected_features = sig_genes.to_list()

        else:
            raise ValueError(
                f"Invalid gene selection method: {self.gene_selection_method}. "
                "Choose 'HVG' or 'Diff'.")
        print(
            f"Selected {len(selected_features)} "
            f"features using {self.gene_selection_method}")

        # Save selected genes/features
        with open(self.features_out_fn, "wb") as f:
            pickle.dump(selected_features, f)

        return selected_features

    def _prepare_scaled_data(self):
        """
        Prepares scaled data for training and testing, fitting scaler only on
        training data.
        Returns:
            x_train, x_test, y_train, y_test
        """
        # Extract training and test data for selected features
        x_train = self.adata[self.train_mask, self.selected_features].X
        x_test = self.adata[self.test_mask, self.selected_features].X

        # Convert to dense arrays if sparse
        if not isinstance(x_train, np.ndarray):
            x_train = x_train.toarray()
        if not isinstance(x_test, np.ndarray):
            x_test = x_test.toarray()

        # Fit scaler on training data, transform both train and test
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        self.scaler = scaler  # Store for later use

        # Store labels
        y_train = self.adata.obs.loc[self.train_mask, 'Label'].values
        y_test = self.adata.obs.loc[self.test_mask, 'Label'].values

        return x_train_scaled, x_test_scaled, y_train, y_test
