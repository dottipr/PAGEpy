#!/usr/bin/env python
# coding: utf-8

"""
Bulk Analysis Script
Converted from Jupyter notebook for server executions

Last updated: 27.08.2025
"""

import logging
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import pandas as pd

from PAGEpy import get_logger, plot_functions, pso, setup_logging, utils
from PAGEpy.dataset_class import GeneExpressionDataset
from PAGEpy.models import AdvancedNN, NNModelConfig, SimpleNN

setup_logging(
    level=logging.INFO,
    log_file="bulk_analysis_output.log",
    console_output=True
)
logger = get_logger(__name__)


def main():
    """Main function to run the bulk analysis pipeline."""

    # Set run ID
    # run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    run_id = "250901_212046"  # e.g., can be set to a previously crashed run ID

    logger.info("Starting Bulk Analysis Pipeline for Run ID '%s'...", run_id)
    logger.info("=" * 50)

    matplotlib.use('Agg')

    # Configure output filenames
    output_dir = "bulk_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_directory = os.path.join(output_dir, f"{run_id}_data")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Initialize CUDA for GPU support
    logger.info("Initializing CUDA...")
    gpu_available = utils.init_tensorflow()

    # Dataset parameters

    logger.info("Creating dataset with Differential Expression Analysis...")

    # Create Dataset
    current_data = GeneExpressionDataset(
        data_dir="../../bulk_data/",
        counts_pattern="count_matrix.mtx",
        barcodes_pattern="sample_names.txt",
        genes_pattern="gene_names.txt",
        metadata_pattern="response_labels.csv",
        gene_selection="Diff",
        pval_cutoff=0.00005,
        pval_correction="benjamini-hochberg",
        features_out_filename=os.path.join(
            data_directory, "feature_set.pkl"),
        train_samples_out_filename=os.path.join(
            data_directory, "train_samples.txt"),
        positive_label="yes"
    )

    # Load selected genes
    genes_path = os.path.join(
        data_directory, "feature_set.pkl")

    with open(genes_path, "rb") as f:
        current_genes = pickle.load(f)
    logger.info("Loaded %d genes", len(current_genes))

    # Set NN model parameters
    config = NNModelConfig(
        report_frequency=1,
        auc_threshold=1,
        learning_rate=0.001
    )

    training_params = {
        'n_epochs': 500,
        'batch_size': 64,
        'seed': 42,
    }

    logger.info("Training initial NN model...")
    logger.info("Training parameters: %s", training_params)

    # Initialize and train initial NN model
    initial_model = AdvancedNN(
        n_input_features=len(current_genes),
        config=config,
    )

    # Train model
    train_history = initial_model.train(
        x_train=current_data.x_train,
        y_train=current_data.y_train,
        x_test=current_data.x_test,
        y_test=current_data.y_test,
        **training_params,
    )

    logger.info("Initial model training completed!")

    # Plot initial model history
    logger.info("Plotting initial model history...")
    plot_functions.plot_model_history(
        model_history=train_history,
        report_frequency=initial_model.config.report_frequency,
        y_train=current_data.y_train,
        y_test=current_data.y_test,
        save_path=os.path.join(
            data_directory, "initial_model_history.png"),
        data_save_path=os.path.join(
            data_directory, "initial_training_metrics.csv")
    )

    # Run binary PSO
    logger.info("Starting binary PSO optimization...")
    logger.info("This may take a while...")

    pso_params = {
        'run_id': run_id,
        'pop_size': 200,
        # 'pop_size': 5,
        'n_generations': 15,
        # 'n_generations': 2,
        'model_class': SimpleNN,
        'w': 1,
        'c1': 2,
        'c2': 1.5,
        'n_reps': 4,
        # 'n_reps': 1,
        'verbose': True,
        'adaptive_metrics': False,
        'output_prefix': data_directory
    }

    logger.info("PSO parameters: %s", pso_params)

    best_solution, best_fitness = pso.run_binary_pso(
        input_data=current_data,
        feature_names=current_genes,
        **pso_params
    )

    logger.info("PSO optimization completed!")
    logger.info("Best fitness: %s", best_fitness)

    # Load and plot PSO results
    logger.info("Loading PSO results and generating plots...")

    try:
        loaded_fitness_scores = pd.read_pickle(
            os.path.join(data_directory, "pso_fitness_scores.pkl"))
        loaded_particle_history = pd.read_pickle(
            os.path.join(data_directory, "pso_particle_history.pkl"))

        # Generate PSO plots
        plot_functions.plot_pso_fitness_evolution(
            fitness_history=loaded_fitness_scores,
            save_path=os.path.join(data_directory, "pso_fitness_evolution.png"))
        plot_functions.plot_population_diversity(
            particle_history=loaded_particle_history,
            save_path=os.path.join(data_directory, "pso_population_diversity.png"))
        plot_functions.plot_feature_selection_frequency(
            particle_history=loaded_particle_history,
            save_path=os.path.join(data_directory, "pso_feature_selection_frequency.png"))

        logger.info("PSO plots generated successfully!")

    except FileNotFoundError as e:
        logger.warning("Warning: Could not load PSO results files: %s", e)
        logger.warning("Proceeding with best_solution from PSO run...")

    # Load PSO selected genes
    logger.info("Loading PSO selected genes...")
    try:
        with open(os.path.join(data_directory, "pso_selected_genes.pkl"), "rb") as f:
            pso_genes = pickle.load(f)
    except FileNotFoundError:
        logger.warning(
            "PSO genes file not found, extracting from best_solution...")
        pso_genes = [item for item, m in zip(
            current_genes, best_solution) if m == 1]
        # Save for future use
        with open(os.path.join(data_directory, "pso_selected_genes.pkl"), "wb") as f:
            pickle.dump(pso_genes, f)

    n_pso_input_features = len(pso_genes)
    logger.info("Number of PSO selected genes: %d", n_pso_input_features)

    # Train improved model with PSO selected features
    logger.info("Training improved model with PSO selected features...")

    improved_model = AdvancedNN(
        n_input_features=n_pso_input_features,
        config=config,
    )

    # Select feature subset and scale data
    x_train, x_test, y_train, y_test = current_data.get_scaled_feature_subset(
        feature_subset=pso_genes
    )

    # Train improved model
    improved_train_history = improved_model.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        **training_params,
    )

    logger.info("Improved model training completed!")

    # Plot improved model history
    logger.info("Plotting improved model history...")
    plot_functions.plot_model_history(
        model_history=improved_train_history,
        report_frequency=improved_model.config.report_frequency,
        y_train=y_train,
        y_test=y_test,
        save_path=os.path.join(
            data_directory, "improved_model_history.png"),
        data_save_path=os.path.join(
            data_directory, "improved_training_metrics.csv")
    )

    logger.info("Analysis pipeline completed successfully!")
    logger.info("=" * 50)

    # Print summary
    logger.info("BULK ANALYSIS SUMMARY:")
    logger.info("- PSO selected features: %d", len(pso_genes))
    logger.info("- Best PSO fitness: %s", best_fitness)
    logger.info("- GPU available: %s", gpu_available)

    logger.info("Generated files in '%s':", data_directory)
    logger.info("- feature_set.pkl")
    logger.info("- train_samples.txt")
    logger.info("- pso_fitness_scores.pkl")
    logger.info("- pso_particle_history.pkl")
    logger.info("- pso_selected_genes.pkl")
    logger.info("- initial_model_history.png")
    logger.info("- improved_model_history.png")
    logger.info("- selected_genes.txt")
    logger.info("- initial_training_metrics.csv")
    logger.info("- improved_training_metrics.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Script interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Error occurred: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
