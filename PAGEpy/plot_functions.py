import logging
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_model_history(
    model_history: Dict[str, List[float]],
    report_frequency: int,
    y_train: list,
    y_test: list,
    save_path: Union[str, Path, None] = None,
    data_save_path: Union[str, Path, None] = None
) -> None:
    """
    Plot model training and validation metrics over epochs.
    Optionally save the values used for plotting to disk as a CSV.

    Args:
        model_history: Dict with lists of metrics per epoch
        report_frequency: Frequency of reporting metrics
        y_train, y_test: Train and test set labels (for chance level)
        save_path: Optional path to save the generated plots
        data_save_path: Optional path to save the plot data as CSV
    """
    # Print maximum metrics
    logger.info(
        "Max metrics\n\tTrain Accuracy: %.2f | Train AUC: %.2f | Test Accuracy: %.2f | Test AUC: %.2f",
        max(model_history['train_accuracy']),
        max(model_history['train_auc']),
        max(model_history['test_accuracy']),
        max(model_history['test_auc'])
    )

    axs = plt.subplots(4, 1, figsize=(12, 8))[1]

    epochs = np.arange(
        1, len(model_history['train_accuracy']) + 1) * report_frequency

    # Prepare data for saving
    plot_data = pd.DataFrame({
        'epoch': epochs,
        'train_accuracy': model_history['train_accuracy'],
        'train_auc': model_history['train_auc'],
        'test_accuracy': model_history['test_accuracy'],
        'test_auc': model_history['test_auc'],
    })
    plot_data['train_chance'] = pd.Series(
        y_train).value_counts(normalize=True).max()
    plot_data['test_chance'] = pd.Series(
        y_test).value_counts(normalize=True).max()

    if data_save_path:
        plot_data.to_csv(data_save_path, index=False)

    # Plot train accuracy
    train_chance = plot_data['train_chance'][0]
    axs[0].plot(epochs, model_history['train_accuracy'],
                label='Training Accuracy', color='blue')
    axs[0].axhline(train_chance, color='black',
                   linestyle='--', label='Chance Level')
    axs[0].set_title('Training set accuracy over epochs (chance level: '
                     f'{train_chance:.3f})')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Training Accuracy')
    axs[0].grid()

    # Plot train auc
    axs[1].plot(epochs, model_history['train_auc'],
                label='Train AUC', color='blue')
    axs[1].set_title('Training set AUC over epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Train AUC')
    axs[1].grid()

    # Plot test accuracy
    test_chance = plot_data['test_chance'][0]
    axs[2].plot(epochs, model_history['test_accuracy'],
                label='Test Accuracy', color='orange')
    axs[2].axhline(test_chance, color='black',
                   linestyle='--', label='Chance Level')
    axs[2].set_title('Test set accuracy over epochs (chance level: '
                     f'{test_chance:.3f})')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Test Accuracy')
    axs[2].grid()

    # Plot test auc
    axs[3].plot(epochs, model_history['test_auc'],
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
