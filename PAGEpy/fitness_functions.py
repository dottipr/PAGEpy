import time

import numpy as np

from PAGEpy import get_logger
from PAGEpy.k_folds_class import KFoldData
from PAGEpy.models import SimpleNN, TrainingConfig

logger = get_logger(__name__)

# TODO: adattare alla nuova implementazione dei K folds!!
# TODO: vedere com'è meglio gestire i kwargs (o simile) per la fitness fct!!


def evaluate_selected_genes_fitness(
    particle: np.ndarray,
    kfolds: KFoldData,
    feature_names: list,
    particle_id: int,
    model_class=None,
    training_params: dict = None,
    training_config: dict = None,
    verbose: bool = True,
    **fitness_kwargs  # Accept additional keyword arguments
):
    """
    Evaluate the fitness of a gene selection (PSO particle) using
    a neural network model and cross-validation.

    This is the core fitness function for the PSO algorithm. It takes
    a binary vector representing which genes to select, trains neural
    networks using 5-fold cross-validation, and returns the average AUC
    as the fitness score.

    Args:
        particle (array-like): Binary vector indicating selected genes (1=selected, 0=not)
        crossval_folds (MultipleFolds): Cross-validation data structure
        all_gene_names (list): Complete list of available gene names
        particle_id (int): Index of current population member (for progress reporting)
        model_class: Model class to instantiate (default: SimpleNN)
        model_params (dict): Parameters for model initialization
        training_params (dict): Parameters for model training
        loud (bool): Whether to print progress messages (default: True)

    Returns:
        float: Fitness score (average AUC across all folds)
               Returns 0 if no genes are selected


    Note: This function is computationally expensive as it trains k
          neural networks per evaluation.
    """
    start_time = time.time()

    # Convert individual to boolean array for gene selection
    selected_gene_names = [
        gene for gene, selected in zip(feature_names, particle) if selected
    ]
    n_selected_genes = len(selected_gene_names)

    # Set default parameters
    if training_config is None:
        # Use PAGEpy's default config
        training_config = TrainingConfig(
            learning_rate=0.001,
            l2_reg=0.2,
            balance_classes=True,
            report_frequency=10,
            auc_threshold=0.999,
        )
    else:
        training_config = TrainingConfig(training_config)

    if model_class is None:
        model_class = SimpleNN

    if training_params is None:
        training_params = {
            'n_epochs': 50,
            'batch_size': 512,
        }

    # Return zero fitness if no genes are selected
    if n_selected_genes == 0:
        logger.warning(
            "Particle %d: No genes selected. Fitness set to 0.",
            particle_id + 1
        )
        return 0.0

    fold_aucs = []

    # Perform k-fold cross-validation
    for fold_id, fold in enumerate(kfolds):
        try:
            x_train, x_test, y_train, y_test = fold.reduce_input_features(
                selected_gene_names)

            # Instantiate and train model
            nn_model = model_class(
                n_input_features=n_selected_genes,
                config=training_config
            )
            nn_model.train(x_train=x_train, y_train=y_train, **training_params)

            # Evaluate model
            fold_auc = nn_model.evaluate(x_test, y_test)
            fold_aucs.append(fold_auc)

            if verbose:
                logger.debug(
                    "Particle %d, Fold %d/%d: AUC = %.4f",
                    particle_id + 1, fold_id + 1, kfolds.n_folds, fold_auc
                )

        except Exception as e:
            logger.error(
                "Particle %d, Fold %d/%d: Error during training/evaluation: %s",
                particle_id + 1, fold_id +
                1, kfolds.n_folds, str(e)
            )
            fold_aucs.append(0.0)  # Assign zero fitness for failed folds

    # Calculate final fitness score as average AUC across folds
    score = np.mean(fold_aucs)
    # score = round(score,3)
    # Round for consistency and to avoid precision issues
    # TODO: check if rounding is necessary of if he's only doing this for printing
    # score = round(float(score), 3)

    end_time = time.time()

    if verbose:
        logger.info(
            "Particle %d: Genes=%d | Mean AUC=%.3f | Time=%.2fs",
            particle_id + 1, n_selected_genes, score, end_time - start_time
        )

    return float(score)
