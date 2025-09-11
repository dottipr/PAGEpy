'''
This module provides fitness evaluation functions for feature selection using
Particle Swarm Optimization (PSO) and cross-validation. The core function
evaluates the fitness of a particle (feature subset) by training a model on
k-fold splits and computing the average AUC score. It supports both custom
neural network models and scikit-learn style estimators.

Functions:
    evaluate_particle_fitness:  Evaluates the fitness of a feature subset using 
                                cross-validation and model training.
    _evaluate_single_fold:      Helper function to train and evaluate a model 
                                on a single fold.

Dependencies:
    - numpy
    - scikit-learn
    - PAGEpy (custom modules for logging, k-fold data, and training configuration)
'''

import time
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from PAGEpy import get_logger
from PAGEpy.k_folds_class import KFoldData
from PAGEpy.models import TrainingConfig

logger = get_logger(__name__)

# TODO: vedere com'è meglio gestire i kwargs (o simile) per la fitness fct!!


def evaluate_particle_fitness(
    particle: np.ndarray,
    kfolds: KFoldData,
    feature_names: list,
    particle_id: int,
    generation_nb: int,  # TODO: remove this
    model_class,
    training_params: Optional[dict] = None,
    training_hyperparameters: Optional[dict] = None,
    loud: bool = True,
) -> float:
    """
    Evaluate the fitness of the PSO particles using a model and
    cross-validation.

    This is the core fitness function for the PSO algorithm. It takes
    a binary vector representing which features to select, trains a model using
    5-fold cross-validation, and returns the average AUC as the fitness score.

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
          models per evaluation.
    """
    start_time = time.time()

    # Convert individual to boolean array for gene selection
    selected_features = [
        feat_name for feat_name, selected in zip(feature_names, particle) if selected
    ]
    n_selected = len(selected_features)

    if n_selected == 0:
        logger.warning(
            "Particle %d: No features selected. Fitness set to 0.",
            particle_id + 1
        )
        return 0.0

    # Set defaults
    training_params = training_params or {}
    if training_hyperparameters is None:
        hyperparameters = TrainingConfig(
            learning_rate=0.001,
            l2_reg=0.2,
            balance_classes=True,
            report_frequency=10,
            auc_threshold=0.999,
        )
    else:
        hyperparameters = TrainingConfig(*training_hyperparameters)

    fold_scores = []

    # Perform k-fold cross-validation
    for fold_id, fold in enumerate(kfolds):
        try:
            x_train, x_test, y_train, y_test = fold.reduce_input_features(
                selected_features)

            # Handle different model types
            avg_score = _evaluate_single_fold(
                model_class, training_params, hyperparameters,
                x_train, y_train, x_test, y_test, n_selected
            )

            fold_scores.append(avg_score)

            if loud:
                logger.debug(
                    "Particle %d, Fold %d/%d: AUC = %.4f",
                    particle_id + 1, fold_id + 1, kfolds.n_folds, avg_score
                )

        except Exception as e:
            logger.error(
                "Particle %d, Fold %d/%d: Error during training/evaluation: %s",
                particle_id + 1, fold_id +
                1, kfolds.n_folds, str(e)
            )
            fold_scores.append(0.0)  # Assign zero fitness for failed folds

    avg_score = np.mean(fold_scores)
    # score = round(score,3)
    # Round for consistency and to avoid precision issues
    # TODO: check if rounding is necessary of if he's only doing this for printing
    # score = round(float(score), 3)

    end_time = time.time()

    if loud:
        logger.info(
            "Particle %d: Genes=%d | Mean AUC=%.3f | Time=%.2fs",
            particle_id + 1, n_selected, avg_score, end_time - start_time
        )

    return float(avg_score)


def _evaluate_single_fold(model_class, training_params, hyperparameters,
                          x_train, y_train, x_test, y_test, n_features):
    """Helper function to evaluate a single fold with any model type."""

    # Check if it's your custom NN (has 'train' method and needs n_input_features)
    if hasattr(model_class, '__name__') and 'NN' in model_class.__name__:
        # Custom NN handling
        model = model_class(
            n_input_features=n_features, config=hyperparameters)
        model.train(x_train=x_train, y_train=y_train, **training_params)
        return model.evaluate(x_test, y_test)

    else:
        # Sklearn-style model handling
        model = model_class(**training_params)
        model.fit(x_train, y_train, **training_params)

        # Get AUC score
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            return roc_auc_score(y_test, y_pred_proba)
        elif hasattr(model, 'decision_function'):
            y_pred_scores = model.decision_function(x_test)
            return roc_auc_score(y_test, y_pred_scores)
        else:
            # Fallback to accuracy
            y_pred = model.predict(x_test)
            return np.mean(y_pred == y_test)
