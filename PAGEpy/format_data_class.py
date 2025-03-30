# format the data for the neural network
import scanpy as sc
import pandas as pd
from scipy.io import mmread
import scipy.sparse
import fnmatch
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

class FormatData:
    def __init__(
        self,
        data_dir = '/your/local/dir/data_folder/',
        test_set_size=0.2,
        random_seed=1,
        hvg_count = 1000,
        pval_cutoff = 0.01,
        gene_selection = 'HVG',
        pval_correction = 'bonferroni'
    ):
        """
        Initializes the RcDataPreparation class with specified parameters.

        Parameters:
        - data_dir (str): Path to the holder containing the neccesary files.
        - test_set_size (float): Fraction of data to be used as a test set (default: 0.2).
        - random_seed (int): Seed for reproducible dataset splits (default: 1).
        - hvg_count (int): number of HVGs for selection
        - gene_selection (str): method of feature selection can either be 'HVG' or 'Diff'
        """

        self.data_dir = data_dir
        self.test_set_size = test_set_size
        self.random_seed = random_seed
        self.hvg_count = hvg_count
        self.pval_cutoff = pval_cutoff
        self.gene_selection = gene_selection
        self.pval_correction = pval_correction
        
        # Initialize placeholders for class attributes
        self.adata = None
        self.counts_df = None
        self.target_variable = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.genes_list = None
        self.selected_genes = None
        self.train_indices = None
        self.test_indices = None
        self.genes = None
        self.barcodes = None
        self.selected_genes = None

        # Automatically execute the data preparation pipeline
        self.construct_and_process_anndata()
        self.encode_labels()
        self.retrieve_counts_df()
        self.retrieve_all_genes()
        self.scale_data()
        self.establish_test_train()

    def construct_and_process_anndata(self):
        """
        Constructs an anndata object.

        Raises:
        - FileNotFoundError: If the specified RDS file does not exist.
        - ValueError: If an error occurs during data extraction or conversion.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Path not found at: {self.data_dir}")

        try:
            relevant_files = os.listdir(self.data_dir)

            matrix_path = self.data_dir + fnmatch.filter(relevant_files, "*counts.mtx")[0]
            barcodes_path = self.data_dir + fnmatch.filter(relevant_files, "*barcodes.txt")[0]
            genes_path = self.data_dir + fnmatch.filter(relevant_files, "*genes.txt")[0]
            target_var_path = self.data_dir + fnmatch.filter(relevant_files, "*target_variable.csv")[0]

            X = mmread(matrix_path).tocsc()  # Convert to Compressed Sparse Column format for efficiency
            self.barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].values  # Read as a NumPy array
            self.genes = pd.read_csv(genes_path, header=None, sep="\t")[0].values  # Read as a NumPy array

            # Create AnnData object
            self.adata = sc.AnnData(X.T)
            self.adata.obs_names = self.barcodes  # Assign cell barcodes
            self.adata.var_names = self.genes  # Assign gene names
            # Load the metadata status as metadata            
            metadata_df = pd.read_csv(target_var_path)
            # Assign the DataFrame to adata.obs
            self.adata.obs = metadata_df
            print("Anndata successfully constructed.")
            # Normalize each cell by total counts (to 10,000 counts per cell)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            # Logarithmize the data
            sc.pp.log1p(self.adata)
            print('Anndata object counts are now normalized.')
        except Exception as e:
            raise ValueError(f"Failed to construct anndata object: {e}")

    def encode_labels(self):
        """
        Encodes the target variable into numerical values.
        """
        self.target_variable = self.adata.obs['Status']
        label_encoder = LabelEncoder()
        self.target_variable = label_encoder.fit_transform(self.target_variable)
        
    def retrieve_counts_df(self):
        # Ensure that the number of rows and columns match
        if len(self.adata.obs_names) != self.adata.X.shape[0]:
            raise ValueError(f"Mismatch between number of cells in adata.obs_names ({len(self.adata.obs_names)}) and rows in adata.X ({self.adata.X.shape[0]})")

        if len(self.adata.var_names) != self.adata.X.shape[1]:
            raise ValueError(f"Mismatch between number of genes in adata.var_names ({len(self.adata.var_names)}) and columns in adata.X ({self.adata.X.shape[1]})")

        # Convert sparse matrix to dense array (if memory allows)
        try:
            dense_matrix = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        except MemoryError:
            print("MemoryError: Sparse matrix will be used instead of dense.")
            dense_matrix = self.adata.X

        # Create DataFrame with genes as columns and barcodes as index
        self.counts_df = pd.DataFrame(dense_matrix, index=self.adata.obs_names, columns=self.adata.var_names)

        # Ensure that rows (barcodes) are labeled correctly with adata.obs_names
        self.counts_df.index = self.adata.obs['Sample']

    def retrieve_all_genes(self):
        self.genes_list = self.adata.var_names.to_list()

    def scale_data(self):
        scaler = MinMaxScaler()
        self.counts_df = scaler.fit_transform(self.counts_df)
        
    def establish_test_train(self):
        """
        Splits the dataset into training and testing sets and then selects HVGs only from the training set.
        """
        indices = np.arange(self.adata.shape[0])

        # Split first
        self.x_train, self.x_test, self.y_train, self.y_test, self.train_indices, self.test_indices = train_test_split(
            self.counts_df, self.target_variable, indices,
            test_size=self.test_set_size, random_state=self.random_seed, 
            stratify=self.target_variable
        )

        # Selects features from only the training set
        adata_train = self.adata[self.train_indices].copy()
        
        if self.gene_selection == 'HVG':
            # Compute HVGs only from the training data
            sc.pp.highly_variable_genes(adata_train, n_top_genes=self.hvg_count, n_bins=100)
            # Get the list of HVGs
            self.selected_genes = adata_train.var.index[adata_train.var['highly_variable']].tolist()
        elif self.gene_selection == 'Diff':
            sc.tl.rank_genes_groups(adata_train,'Status',method='t-test', key_added = "t-test", corr_method = self.pval_correction)
            sig_genes = sc.get.rank_genes_groups_df(adata_train, group = self.adata.obs['Status'][0], key='t-test', pval_cutoff=self.pval_cutoff)['names']
            self.selected_genes = sig_genes.to_list()  
        # Save selected_genes
        with open("feature_set.pkl", "wb") as f:
            pickle.dump(self.selected_genes, f)
        print(f"The total length of the genes list or feature set is: {len(self.selected_genes)}.")

        # Save training sample names
        with open('train_samples.txt', 'w') as f:
            training_samples = pd.Series(self.barcodes[self.train_indices])
            for name in training_samples.tolist():
                f.write(f"{name}\n")
