#!/usr/bin/env python
# coding: utf-8

"""
Single Cell Analysis Script
Converted from Jupyter notebook for server execution

Last updated: 26.08.2025
"""

import os
import pickle
import sys
from datetime import datetime

import matplotlib
import pandas as pd

from PAGEpy import plot_functions, pso, utils
from PAGEpy.dataset_class import GeneExpressionDataset
from PAGEpy.models import AdvancedNN, TrainingConfig


def main():
    """Main function to run the single cell analysis pipeline."""

    print("Starting Single Cell Analysis Pipeline...")
    print("=" * 50)

    matplotlib.use('Agg')

    # Configure output filenames
    # Can be set to a previously crashed run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "single_cell"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_prefix = os.path.join(output_dir, f"{run_id}_")

    data_directory = output_prefix + "data"
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Initialize CUDA for GPU support
    print("Initializing CUDA...")
    gpu_available = utils.init_tensorflow()

    # Dataset parameters
    n_hvg_input_features = 2000

    print(f"\nCreating dataset with {n_hvg_input_features} HVG features...")

    # Create Dataset
    current_data = GeneExpressionDataset(
        data_dir="../../HIVdata/",
        counts_pattern="*counts.mtx",
        barcodes_pattern="*barcodes.txt",
        genes_pattern="*genes.txt",
        metadata_pattern="*infection_status.csv",
        gene_selection="HVG",
        pval_correction="benjamini-hochberg",
        hvg_count=n_hvg_input_features,
        features_out_filename=os.path.join(
            data_directory, "feature_set.pkl"),
        train_samples_out_filename=os.path.join(
            data_directory, "train_samples.txt"),
    )

    # Load selected genes
    genes_path = os.path.join(
        data_directory, "feature_set.pkl")

    with open(genes_path, "rb") as f:
        current_genes = pickle.load(f)
    print(f"Loaded {len(current_genes)} genes")

    # Set NN model parameters
    config = TrainingConfig(
        report_frequency=1,
        auc_threshold=1,
        learning_rate=0.001
    )

    training_params = {
        'n_epochs': 500,
        'batch_size': 512,
        'seed': 42,
    }

    print("\nTraining initial NN model...")
    print(f"Training parameters: {training_params}")

    # Initialize and train initial NN model
    initial_model = AdvancedNN(
        n_input_features=n_hvg_input_features,
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

    print("Initial model training completed!")

    # Plot initial model history
    print("Plotting initial model history...")
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
    print("\nStarting binary PSO optimization...")
    print("This may take a while...\n")

    pso_params = {
        'pop_size': 200,
        # 'pop_size': 5,
        'n_generations': 15,
        # 'n_generations': 2,
        'w': 1,
        'c1': 2,
        'c2': 1.5,
        'n_reps': 4,
        # 'n_reps': 1,
        'verbose': True,
        'adaptive_metrics': False,
        'output_prefix': data_directory
    }

    print(f"PSO parameters: {pso_params}")

    best_solution, best_fitness = pso.run_binary_pso(
        input_data=current_data,
        feature_names=current_genes,
        **pso_params
    )

    print("PSO optimization completed!")
    print(f"Best fitness: {best_fitness}")

    # Load and plot PSO results
    print("\nLoading PSO results and generating plots...")

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

        print("PSO plots generated successfully!")

    except FileNotFoundError as e:
        print(f"Warning: Could not load PSO results files: {e}")
        print("Proceeding with best_solution from PSO run...")

    # Load PSO selected genes
    print("\nLoading PSO selected genes...")
    try:
        with open(os.path.join(data_directory, "pso_selected_genes.pkl"), "rb") as f:
            pso_genes = pickle.load(f)
    except FileNotFoundError:
        print("PSO genes file not found, extracting from best_solution...")
        pso_genes = [item for item, m in zip(
            current_genes, best_solution) if m == 1]
        # Save for future use
        with open(output_prefix + "pso_selected_genes.pkl", "wb") as f:
            pickle.dump(pso_genes, f)

    n_pso_input_features = len(pso_genes)
    print(f"Number of PSO selected genes: {n_pso_input_features}")

    # Train improved model with PSO selected features
    print("\nTraining improved model with PSO selected features...")

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

    print("Improved model training completed!")

    # Plot improved model history
    print("Plotting improved model history...")
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

    print("\nAnalysis pipeline completed successfully!")
    print("=" * 50)

    # Print summary
    print("\nSINGLE CELL ANALYSIS SUMMARY:")
    print(f"- Initial features: {n_hvg_input_features}")
    print(f"- PSO selected features: {len(pso_genes)}")
    print(f"- Best PSO fitness: {best_fitness}")
    print(f"- GPU available: {gpu_available}")

    print(f"\nGenerated files in '{data_directory}':")
    print("- feature_set.pkl")
    print("- train_samples.txt")
    print("- pso_fitness_scores.pkl")
    print("- pso_particle_history.pkl")
    print("- pso_selected_genes.pkl")
    print("- initial_model_history.png")
    print("- improved_model_history.png")
    print("- selected_genes.txt")
    print("- initial_training_metrics.csv")
    print("- improved_training_metrics.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
