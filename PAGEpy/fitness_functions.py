import time

import numpy as np

from PAGEpy.k_folds_class import KFoldData
from PAGEpy.models import SimpleNN, TrainingConfig

# TODO: adattare alla nuova implementazione dei K folds!!
# TODO: vedere com'è meglio gestire i kwargs (o simile) per la fitness fct!!


def evaluate_selected_genes_fitness(
    particle: np.ndarray,
    kfolds: KFoldData,
    feature_names: list,
    particle_id: int,
    generation_nb: int,
    model_class=None,
    training_params: dict = None,
    training_config: dict = None,
    loud: bool = True,
    **fitness_kwargs  # Accept additional keyword arguments
):
    """
    Evaluate the fitness of a gene selection (PSO particle) using
    cross-validation.

    This is the core fitness function for the PSO algorithm. It takes
    a binary vector representing which genes to select, trains neural
    networks using 5-fold cross-validation, and returns the average AUC
    as the fitness score.

    Args:
        particle (array-like): Binary vector indicating selected genes (1=selected, 0=not)
        crossval_folds (MultipleFolds): Cross-validation data structure
        all_gene_names (list): Complete list of available gene names
        particle_id (int): Index of current population member (for progress reporting)
        generation_nb (int): Current generation number (for progress reporting)
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
        if loud:
            print(
                f"No genes selected for particle {particle_id+1}, generation {generation_nb}")
        return 0.0

    # Progress reporting
    if loud:
        print(f"Training particle {particle_id+1}, generation {generation_nb} "
              f"with {n_selected_genes} genes")

    fold_aucs = []

    # Perform k-fold cross-validation
    for fold_id, fold in enumerate(kfolds):
        try:
            # TODO: check se funziona con nuova implementazione dei folds!!
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

            if loud:
                print(
                    f"  Fold {fold_id + 1}/{kfolds.n_folds}: AUC = {fold_auc:.4f}")

        except Exception as e:
            if loud:
                print(f"  Error in fold {fold_id + 1}: {str(e)}")
            fold_aucs.append(0.0)  # Assign zero fitness for failed folds

    if loud:
        print(f"All folds completed for particle {particle_id+1}, "
              f"generation {generation_nb}")

    # Calculate final fitness score as average AUC across folds
    score = np.mean(fold_aucs)
    # score = round(score,3)
    # Round for consistency and to avoid precision issues
    # TODO: check if rounding is necessary of if he's only doing this for printing
    # score = round(float(score), 3)

    end_time = time.time()

    if loud:
        print(f"Average final test AUC value: {round(score, 3)}")
        print(f"Total time: {end_time - start_time} seconds")

    return float(score)
