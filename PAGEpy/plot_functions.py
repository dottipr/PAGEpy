from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PAGEpy.format_data_class import GeneExpressionDataset
from PAGEpy.models import SimpleNN


def plot_model_history(
    model: SimpleNN,
    data: GeneExpressionDataset,
    save_path: Union[str, Path, None] = None
) -> None:
    """
    Plot model training and validation metrics over epochs.

    Args:
        model: Trained neural network model with history attributes
        data: Data fold containing train and test sets and is used to 
              calculate chance levels (# positive samples / # samples)
        save_path: Optional path to save the generated plots
    """
    # Print maximum metrics
    print(f"Max train accuracy: {max(model.train_accuracy_list):.2f}")
    print(f"Max train AUC: {max(model.train_auc_list):.2f}")
    print(f"Max test accuracy: {max(model.test_accuracy_list):.2f}")
    print(f"Max test AUC: {max(model.test_auc_list):.2f}")

    axs = plt.subplots(4, 1, figsize=(12, 8))[1]

    epochs = np.arange(1, len(model.train_accuracy_list) +
                       1) * model.report_frequency

    # Plot train accuracy
    train_chance = pd.Series(data.y_train).value_counts(normalize=True).max()
    axs[0].plot(epochs, model.train_accuracy_list,
                label='Training Accuracy', color='blue')
    axs[0].axhline(train_chance, color='black',
                   linestyle='--', label='Chance Level')
    axs[0].set_title('Training set accuracy over epochs (chance level: '
                     f'{train_chance:.3f})')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Training Accuracy')
    axs[0].grid()

    # Plot train auc
    axs[1].plot(epochs, model.train_auc_list,
                label='Train AUC', color='blue')
    axs[1].set_title('Training set AUC over epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Train AUC')
    axs[1].grid()

    # Plot test accuracy
    test_chance = pd.Series(data.y_test).value_counts(normalize=True).max()
    axs[2].plot(epochs, model.test_accuracy_list,
                label='Test Accuracy', color='orange')
    axs[2].axhline(test_chance, color='black',
                   linestyle='--', label='Chance Level')
    axs[2].set_title('Test set accuracy over epochs (chance level: '
                     f'{test_chance:.3f})')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Test Accuracy')
    axs[2].grid()

    # Plot test auc
    axs[3].plot(epochs, model.test_auc_list,
                label='Test AUC', color='orange')
    axs[3].set_title('Test set AUC over epochs')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Test AUC')
    axs[3].grid()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pso_fitness_evolution(
        fitness_history: List[Dict[str, float]],
        save_path: Union[str, Path, None] = None
) -> None:
    """
    Plot PSO fitness evolution over generations.

    Args:
        fitness_history: List of dictionaries containing fitness scores
        save_path: Optional path to save the plot
    """
    # Convert fitness history to DataFrame
    df = pd.DataFrame(fitness_history)

    # Calculate statistics
    mean_fitness = df.mean(axis=1)
    std_fitness = df.std(axis=1)
    max_fitness = df.max(axis=1)
    min_fitness = df.min(axis=1)

    plt.figure(figsize=(10, 6))
    generations = range(len(fitness_history))

    # Plot mean fitness with std deviation band
    plt.plot(generations, mean_fitness, 'b-', label='Mean Fitness')
    plt.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.2,
        color='b'
    )

    # Plot max and min fitness
    plt.plot(generations, max_fitness, 'g--', label='Best Fitness')
    plt.plot(generations, min_fitness, 'r--', label='Worst Fitness')

    plt.xlabel('Generation')
    plt.ylabel('Fitness Score (AUC)')
    plt.title('PSO Optimization Progress')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_population_diversity(
        particle_history: dict,
        save_path: Union[str, Path, None] = None,
) -> None:
    """
    Plot population diversity metrics (Hamming distance) over generations.

    Args:
        particle_history: Dictionary of particle positions by generation
        save_path: Optional path to save the plot
    """
    generations = list(particle_history.keys())
    avg_hamming = []

    for gen in generations:
        population = particle_history[gen]
        distances = []

        # Calculate pairwise Hamming distances
        for i, particle_i in enumerate(population):
            for _, particle_j in enumerate(population[i + 1:], start=i + 1):
                dist = np.sum(particle_i != particle_j)
                distances.append(dist)

        avg_hamming.append(np.mean(distances))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_hamming, 'b-', label='Average Hamming Distance')
    plt.xlabel('Generation')
    plt.ylabel('Average Hamming Distance')
    plt.title('Population Diversity Over Time')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_selection_frequency(
        particle_history: dict,
        save_path: Union[str, Path, None] = None,
) -> None:
    """
    Plot feature selection frequency in first vs last generation.
    X-axis (Feature Index): Shows features sorted by their selection frequency
    Y-axis (Selection Frequency): Range from 0.0 to 1.0 representing how often 
                                  a feature is selected

    If the last generation contains many values near 1.0 or 0.0:
        Clear feature selection, the algorithm is confident
    If the last generation contains many values near 0.5:
        Uncertain feature selection, the algorithm is undecided

    Args:
        particle_history: Dictionary of particle positions by generation
        save_path: Optional path to save the plot
    """
    first_gen = particle_history[0]
    last_gen = particle_history[len(particle_history) - 1]

    # Calculate selection frequencies
    first_freq = np.mean(first_gen, axis=0)
    last_freq = np.mean(last_gen, axis=0)

    # Sort frequencies for better visualization
    sorted_first = np.sort(first_freq)
    sorted_last = np.sort(last_freq)

    plt.figure(figsize=(10, 6))
    plt.plot(
        sorted_first,
        marker='o',
        linestyle='--',
        color='blue',
        label='First Generation')
    plt.plot(
        sorted_last,
        marker='s',
        linestyle='--',
        color='red',
        label='Last Generation')
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Selection Frequency')
    plt.title('Feature Selection Frequency Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path)
    plt.show()
