# PAGEpy - Predictive Analysis of Gene Expression with Python

## Overview
PAGEpy (Predictive Analysis of Gene Expression in Python) is compatible with both single-cell and bulk RNA sequencing datasets. It requires four input files placed in a single directory:

1. A counts matrix, where genes are rows and cells or samples are columns.
2. A text file containing the list of all gene names.
3. A text file containing the list of all sample/barcode names.
4. A CSV file with the target variable of interest.

This repository provides code to format these datasets, split them into training and test groups, and select highly variable genes (HVGs) to be used as features for a neural network. The network is trained using a custom protocol aimed at minimizing overfitting and training set memorization.

Several customization options are available for different aspects of the process. Additionally, the repository includes a Particle Swarm Optimization (PSO) pipeline to improve the feature set. The PSO pipeline generates many randomized subsets of the initial set of HVGs and iteratively optimizes the population using PSO. Population members are evaluated with simplified neural network architectures over a small number of epochs. A variety of tuning parameters are provided for customization.

The end result of the PSO pipeline is a set of features, based solely on the training dataset, that could potentially improve the model’s generalizability.

The codebase also includes various plotting scripts for evaluating the model's performance.

## Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Project Organization](#Project-Organization)
- [Function and class descriptions](#Function-and-class-descriptions)
- [Contributing](#Contributing)
- [License](#License)
- [Contact](#Contact)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sean-otoole/PAGEpy.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your_path/PAGEpy
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To get started, first open your script or notebook and import the necessary packages:
```python
import PAGEpy_plot
import pso
from format_data_class import FormatData
from pred_ann_model import PredAnnModel
import pickle
import pandas as pd
import PAGEpy_utils

PAGEpy_utils.init_cuda()
```
This ensures that all required packages are imported and enables GPU memory growth.

### Formatting the Data
Next, you need to format your data so it can be passed to the relevant classes and functions within the PAGEpy codebase.
```python
current_data = FormatData(data_dir = '/home/your_data_dir/',
        test_set_size=0.2,
        random_seed=1,
        hvg_count = 1000)
```
This instantiates the FormatData class, which requires a directory containing the following four files:

1. A counts matrix, where genes are rows and cells/samples are columns.
2. A text file containing the list of all gene names.
3. A text file containing the list of all sample/barcode names.
4. A CSV file with the target variable of interest.

If you're working with a sparse matrix of class `dgCMatrix` in R, you can easily produce the required file types either in R or using an R API within Python.
```R
library(Matrix)

counts <- readRDS('sparse.RNA.counts.rds')
writeMM(counts, "sparse.RNA.counts.mtx")

write.table(rownames(counts), "genes.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
write.table(colnames(counts), "barcodes.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
```

The `FormatData` class will automatically generate a list of genes for training the model, which is saved in the local directory as `hvgs.pkl`. This gene list is also available within the `FormatData` object. It’s preferred to load it from the file like this:

```python
genes_path = '/home/your_path/hvgs.pkl'

with open(genes_path, 'rb') as f:
    current_genes = pickle.load(f)
```

### Training the Model

Now, you can pass the genes list and `FormatData` object to the `PredAnnModel` class.

```Python
current_model = PredAnnModel(current_data,current_genes,num_epochs=50)
```

The model offers many customization options, but only the input data, feature list, and number of epochs are required arguments. For a complete list of optional input arguments, refer to the class docstring (e.g., `help(PredAnnModel)`). Many of these options are related to regularization techniques.

Several features make this model effective:

1. The learning rate adjusts dynamically based on performance metrics during training.
2. A percentage of the training data is hidden during each epoch to reduce overfitting and prevent memorization.
3. Target variable balancing is applied during the mini-batch training process.
4. The input layer automatically scales based on the number of features passed to the model.

Once the class is instantiated, the model will provide updates on training progress every 10 epochs. For example, here is the output after epoch 40:

```console
Epoch 40, Avg Outcome Loss: 0.3833, Train AUC: 0.9137, Train Accuracy: 0.8339, Test AUC: 0.8909, Test Accuracy: 0.8234
```
### Evaluating the Model
After training the model, you can view its progress and metrics by running:

```python
PAGEpy_plot.evaluate_model(current_model, current_data)
```

This will output:

```console
max train accuracy: 0.86
max train auc: 0.93
max test accuracy: 0.84
max test auc: 0.91
```
![example_training_course.png](example_images/example_training_course.png)

In this case, the model is predicting whether a cell is infected with HIV, using only the endogenous transcriptome.

### Feature Set Optimization

If you're not entirely satisfied with the model's performance, you can adjust various parameters and increase the number of epochs. You can also expand or narrow the feature set. Alternatively, you can use the Particle Swarm Optimization (PSO) algorithm included in PAGEpy to optimize the feature set.

You can execute the PSO algorithm as follows:
```python
best_solution, best_fitness = pso.binary_pso(current_genes, current_data, 200, 20)
```

The `binary_pso` function takes the current genes and splits the data into training and test sets. It randomly selects features for a specified number of population members (in this case, n=200). It then trains 200 models, each using one of the selected feature sets. The model architecture is similar to `PredAnnModel`, but with a simpler training regimen and fewer regularization elements. The models are trained for only 10 epochs. After training, the performance of each population member is used to decide whether features should be mutated. The parameters `C1` and `C2` control how individual and group performance influence these changes, while the momentum term `W` determines the likelihood of a feature change. After each generation (in this case, 20), the population members are updated. If the algorithm works well, there should be an improvement in the average and maximum test AUC values.

Since this algorithm can take a long time to run, it’s helpful to monitor its progress. The `binary_pso` function will produce two local files that you can load and use to track progress:

```python
pso_df = pd.read_pickle("pso_df.pkl")
pso_dict = pd.read_pickle("pso_dict.pkl")

PAGEpy_plot.plot_pso_row_averages(pso_df)
PAGEpy_plot.plot_hamming_distance(pso_dict)
PAGEpy_plot.plot_sorted_frequencies(pso_dict, pso_df)
```
![example_feature_set_performance_across_pso_generations.png](example_images/example_feature_set_performance_across_pso_generations.png)
- plot_pso_row_averages will show how the population is improving over time.

![example_hamming_distance_plot.png](example_images/example_hamming_distance_plot.png)
- plot_hamming_distance will track the average Hamming distance between population members, demonstrating the degree of similarity.

![example_features_frequencies_plot.png](example_images/example_features_frequencies_plot.png)
- plot_sorted_frequencies will show the proportional representation of features in the first and latest generations.

The output of the PSO will return the best performing feature set as well as its assocaited score. Additionally, the best performing feature set will be written within the local directory as such: `pso_genes_result.pkl`.

Subsequently, you can use the optimized feature set to then retrain the model and potentially produce an improved score with regards to the Test set AUC value as such:

```python
with open('pso_genes_result.pkl', 'wb') as f:
    pickle.dump(pso_genes, f)
      
new_model = PredAnnModel(current_data,pso_genes,num_epochs=50)
ann_plot.evaluate_model(new_model, current_data)
```

## Project Organization

```
📂 src/                           # Source code for the PAGEpy project
├── 📄 PAGEpy_plot.py             # Contains various functions for plotting the data and tracking progress
├── 📄 format_data_class.py       # The FormatData class takes expression data and a target variable to instantiate an object suitable for PSO and training a deep neural network
├── 📄 multiple_folds_class.py    # The MultipleFolds class uses the FormatData class as input to generate multiple folds (default = 5) for cross validation
├── 📄 indvidual_fold_class.py    # The IndividualFold class generates a single fold which can than be passed directly to the PredAnnModel class
├── 📄 pred_ann_model.py          # Given either the FormatData or IndividualFold the PredAnnModel class instnatiates and trains a deep neural network for target variable prediction
├── 📄 pso.py                     # Contains a series of functions for a particle swarm optimzation algoriwthm for feature selection
├── 📄 PAGEpy_utils.py            # Contains various helper functions
📂 example_images/                # Contains example images for the readme file
📄 README.md                      # Project description and repository guide
📄 LICENSE                        # MIT license
```

## Function and class descriptions

### :red_circle: `FormatData` class

A class for preparing and formatting gene expression data for machine learning pipelines.

**Initialization**

```python
FormatData(
    data_dir='/home/input_data_folder/'',
    test_set_size=0.2,
    random_seed=1,
    hvg_count=1000,
    pval_cutoff=0.01,
    gene_selection='HVG',
    pval_correction='bonferroni'
)
```

**Parameters**
- `data_dir` (str, default='/home/input_data_folder/'): Path to the directory containing necessary input files.
- `test_set_size` (float, default=0.2): Fraction of the dataset to be used as the test set.
- `random_seed` (int, default=1): Seed for reproducible dataset splits.
- `hvg_count` (int, default=1000): Number of highly variable genes (HVGs) to select.
- `pval_cutoff` (float, default=0.01): Significance threshold for gene selection when using differential expression analysis.
- `gene_selection` (str, default='HVG'): Method of feature selection, can be either: 'HVG' (Highly Variable Genes) or  'Diff' (Differential Expression)
- `pval_correction` (str, default='bonferroni'): Method used for multiple hypothesis testing correction.

**Attributes**
  
- `adata` (AnnData): Stores single-cell expression data.
- `counts_df` (pd.DataFrame): DataFrame of raw count data.
- `target_variable` (array-like): Labels or metadata used for classification.
- `x_train`, `x_test` (pd.DataFrame): Training and test feature matrices.
- `y_train`, `y_test` (array-like): Corresponding training and test labels.
- `genes_list` (list): List of all available genes in the dataset.
- `selected_genes` (list): Subset of genes chosen through HVG or differential expression analysis.
- `train_indices`, `test_indices` (array-like): Indices of samples assigned to training and test sets.
- `genes` (list): Processed gene identifiers.
- `barcodes` (list): Cell barcode identifiers.
- `selected_genes` (list): Genes selected through differential expression testing or highly variable gene identification

**Methods (Automatically Called)**
- `self.construct_and_process_anndata()`: Constructs the anndata object.
- `self.encode_labels()`: Binarizes the target variable for training the model.
- `self.retrieve_counts_df()`: Constructs a counts data frame.
- `self.retrieve_all_genes()`: Attaches the list of all genes/features to the FormatData object.
- `self.scale_data()`: Scales and centers the data.
- `self.establish_test_train()`: Splits the data into test and train sets and performs feature selection using only the training set.

**Example Usage**
```python
from mymodule import FormatData

# Initialize with custom parameters
data_prep = FormatData(
    data_dir='/path/to/data',
    test_set_size=0.20,
    hvg_count=1500,
    gene_selection='HVG',
    pval_correction='Bonferroni'
)

# Access selected genes
print(data_prep.selected_genes)
```

### :red_circle: `PredAnnModel` class

**Initialization**
```python
PredAnnModel(
    input_data,
    current_genes,
    learning_rate=0.01,
    dropout_rate=0.3,
    balance=True,
    l2_reg=0.2,
    batch_size=64,
    num_epochs=5000,
    report_frequency=1,
    auc_threshold=0.95,
    clipnorm=2.0,
    simplify_categories=True,
    holdout_size=0.5,
    multiplier=3,
    auc_thresholds=[0.6, 0.7, 0.8, 0.85, 0.88, 0.89, 0.90, 0.91, 0.92],
    lr_dict={
        0.6:  0.005,
        0.7:  0.001,
        0.8:  0.0005,
        0.85: 0.0005,
        0.88: 0.0005,
        0.89: 0.0005,
        0.9:  0.0005,
        0.91: 0.0005,
        0.92: 0.0005
    }
)
```

**Paramters**

- `input_data` (FormatData object): Processed dataset used for training, containing gene expression values and labels.
- `current_genes` (list): A non-empty list of genes used as model features.
- `learning_rate` (float, default=0.01): Initial learning rate for the model.
- `dropout_rate` (float, default=0.3): Dropout rate to prevent overfitting.
- `balance` (bool, default=True): Whether to balance technology and outcome variables during training.
- `l2_reg` (float, default=0.2): Strength of L2 regularization.
- `batch_size` (int, default=64): Number of samples per training batch.
- `num_epochs` (int, default=5000): Maximum number of training epochs.
- `report_frequency` (int, default=1): Frequency of logging model performance metrics.
- `auc_threshold` (float, default=0.95): AUC threshold for early stopping.
- `clipnorm` (float, default=2.0): Maximum gradient norm to prevent exploding gradients.
- `simplify_categories` (bool, default=True): Whether to simplify data categories before training.
- `holdout_size` (float, default=0.5): Fraction of data withheld for evaluation.
- `multiplier` (int, default=3): Scaling factor for the number of nodes in most layers of the neural network.
- `auc_thresholds` (list, default=[0.6, 0.7, 0.8, 0.85, 0.88, 0.89, 0.90, 0.91, 0.92]): AUC values at which the learning rate is adjusted.
- `lr_dict` (dict): Dictionary mapping AUC thresholds to corresponding learning rates.

**Attributes**
After initialization, the class contains the following attributes:
- `outcome_classifier` (keras.Model): The deep learning model for classification.
- `test_accuracy_list`, `train_accuracy_list` (list): Accuracy metrics collected during training.
- `test_auc_list`, `train_auc_list` (list): AUC values for test and training sets.
- `current_epoch_list` (list): Tracks epochs where metrics were logged.

**Methods (Automatically Called)**
- `set_mixed_precision()`: Enables mixed-precision training for improved performance.
- `subset_input_data()`: Filters input data based on selected genes.
- `build_outcome_classifier()`: Constructs the ANN model.
- `train_the_model()`: Runs the training process.

**Example Usage**
```python
from mymodule import PredAnnModel, FormatData

# Load and process input data
data_prep = FormatData(data_dir='/path/to/data')

# Define a list of genes to use
selected_genes = ['GeneA', 'GeneB', 'GeneC']

# Initialize and train the model
model = PredAnnModel(input_data=data_prep, current_genes=selected_genes)

# Access model performance metrics
print(model.test_auc_list)
```

<br>

### :red_circle: `binary_pso` function

Performs feature selection using a Binary Particle Swarm Optimization (PSO) algorithm to optimize a classification model based on gene expression data.

**Positional arguments**
- current_genes (list): A list of gene names considered for feature selection.
- current_data (pd.DataFrame): A DataFrame containing gene expression values with samples as rows and genes as columns.
- POP_SIZE (int): The number of particles (candidate solutions) in the swarm.
- N_GENERATIONS (int): The number of iterations for the PSO algorithm.

**Keyword arguments**
  
- W (float, optional, default=1): Inertia weight controlling the influence of previous velocity on the new velocity.
- C1 (float, optional, default=2): Cognitive coefficient influencing how much a particle follows its personal best position.
- C2 (float, optional, default=1.5): Social coefficient influencing how much a particle follows the global best position.
- reps (int, optional, default=4): The number of times each feature set is evaluated to account for variability.
- frequent_reporting (bool, optional, default=False): If True, logs intermediate results more frequently.
- adaptive_metrics (bool, optional, default=False): If True, dynamically adjusts evaluation criteria based on observed performance trends.

**Returns**

- best_solution (list): The best performing subset of genes selected.
- best_fitness (float): The highest achieved evaluation metric (e.g., AUC).

**Example Usage**
```python
best_solution, best_fitness = pso.binary_pso(current_genes, current_data, 100, 20)
```

### :bar_chart: Plotting Functions

Several functions are available with the package:
evaluate_model(input_model, input_data)
Plots model training and evaluation metrics over epochs:

Training and test accuracy with chance level baselines.

Training and test AUC scores.

`plot_pso_row_averages(df)`
- Visualizes the performance of feature sets across generations:
Plots the mean, max, min, and standard deviation of AUC values per generation.

`plot_hamming_distance(input_dict)`
- Tracks diversity in feature sets over generations:
Computes and plots the average Hamming distance among feature sets in each generation.

`plot_sorted_frequencies(loaded_dict, loaded_df)`
- Compares feature selection frequencies between the first and latest generation:
Sorts and normalizes the frequency of selected features (1s) and plots the changes over time.

**Example Usage**
```python
import pandas as pd
import PAGEpy_plot

pso_df = pd.read_pickle("pso_df.pkl")
pso_dict = pd.read_pickle("pso_dict.pkl")
PAGEpy_plot.plot_pso_row_averages(pso_df)

PAGEpy_plot.plot_hamming_distance(pso_dict)

PAGEpy_plot.plot_sorted_frequencies(pso_dict,pso_df)
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, reach out to:
- **Sean O'Toole** - [sean.otoole.bio@gmail.com](mailto:your-email@example.com)
- GitHub: [sean-otoole](https://github.com/your-username)

