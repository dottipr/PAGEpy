# PACE - Predictive Analysis of Cell Expression

## Overview
PACE (Predictive Analysis of Cell Expression) is compatible with both single-cell and bulk RNA sequencing datasets. It requires four input files placed in a single-cell folder:

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
- [Configuration](#Configuration)
- [Contributing](#Contributing)
- [License](#License)
- [Contact](#Contact)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sean-otoole/PACE.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your_path/PACE
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
First open your script or notebook with the neccesary imports. This includes the all of the 
```python
import pace_plot
import pso
from format_data_class import FormatData
from pred_ann_model import PredAnnModel
import pickle
import pandas as pd
import pace_utils

pace_utils.init_cuda()
```

This will ensure you have the nccesary packages imported as well as enable memory growth for the gpu.

Next you want to format your data so that it can be passed to other classes and functions within the PACE codebase.

```python
current_data = FormatData(data_dir = '/home/your_data_dir/',
        test_set_size=0.2,
        random_seed=1,
        hvg_count = 1000)
```

This instantiate the FormatData class that requires a folder container the four file types discussed in the overview and they are:

1. A counts matrix, where genes are rows and cells or samples are columns.
2. A text file containing the list of all gene names.
3. A text file containing the list of all sample/barcode names.
4. A CSV file with the target variable of interest.

If you're working with a sparse Matrix of class "dgCMatrix" in R one can easily produce the file types that are required either in R or using an R API within python.

```R
library(Matrix)

counts <- readRDS('sparse.RNA.counts.rds')
writeMM(counts, "sparse.RNA.counts.mtx")

write.table(rownames(counts), "genes.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
write.table(colnames(counts), "barcodes.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
```

The FormatData class will automatiically produce a list of genes that will be used for training the model. This list is saved within the local directory and entitled 'hvgs.pkl. This genes list is also present within the FormatData class object. My preference is to load it from a file as such:

```python
genes_path = '/home/your_path/hvgs.pkl'

with open(genes_path, 'rb') as f:
    current_genes = pickle.load(f)
```

Now you can pass your genes list and FormatData object to the PredAnnModel class. 

```Python
current_model = PredAnnModel(current_data,current_genes,num_epochs=50)
```

The model offers a variety of customization options; however, only the input data, features list, and the number of epochs are required arguments. For a full list of potential input arguments, refer to the class docstring (e.g., help(PredAnnModel)). Many of these options relate to regularization techniques.

Several features make this model effective:

1. The learning rate adjusts dynamically based on performance metrics during training.

2. A percentage of the training data is hidden during each epoch to reduce overfitting and prevent memorization of the training set.

3. Target variable balancing is applied during the mini-batch training process.

4. The input layer automatically scales to accommodate the number of features passed to the model.

Once the class is instantiated, the model will provide updates on training progress every 10 epochs.

For example, here is the output after epoch 40:

```console
Epoch 40, Avg Outcome Loss: 0.3833, Train AUC: 0.9137, Train Accuracy: 0.8339, Test AUC: 0.8909, Test Accuracy: 0.8234
```

## Project Organization

```
📂 src/                           # Source code for the PACE project
├── 📄 pace_plot.py                # Contains various functions for plotting the data and tracking progress
├── 📄 format_data_class.py       # The FormatData class takes expression data and a target variable to instantiate an object suitable for PSO and training a deep neural network
├── 📄 multiple_folds_class.py    # The MultipleFolds class uses the FormatData class as input to generate multiple folds (default = 5) for cross validation
├── 📄 indvidual_fold_class.py    # The IndividualFold class generates a single fold which can than be passed directly to the PredAnnModel class
├── 📄 pred_ann_model.py          # Given either the FormatData or IndividualFold the PredAnnModel class instnatiates and trains a deep neural network for target variable prediction
├── 📄 pso.py                     # Contains a series of functions for a particle swarm optimzation algoriwthm for feature selection
├── 📄 pace_utils.py              # Contains various helper functions
📂 example notebooks/             # Example python notebooks for code easy to understand exuection of the code
├── 📄 example_pipeline.ipynb     # An example notebook for running the pipeline
├── 📄 example_pso_check.ipynb    # An example notebook for checking the progress of the pipeline
📂 example_images/                # Contains example images for the readme file
📄 README.md                      # Project description and repository guide
📄 LICENSE                        # MIT license
📄 requirements.txt               # project depdencies

```

## Configuration
Explain any necessary environment variables or configuration settings.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, reach out to:
- **Sean O'Toole** - [sean.otoole.bio@gmail.com](mailto:your-email@example.com)
- GitHub: [sean-otoole](https://github.com/your-username)

