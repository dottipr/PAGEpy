'''
Modular, class-based code for binary PSO (Particle Swarm Optimization)
applied to gene selection.

This module implements a binary PSO algorithm for feature selection in
genomic data, where each particle represents a binary vector indicating
which genes to include in a neural network classifier. The fitness of
each particle is evaluated through cross-validation performance of the
resulting gene subset.

Main Components:
- Config: Configuration parameters for PSO and neural network training
- SimpleNNModel: Neural network classifier for evaluating gene subsets
- BinaryPSO: PSO algorithm implementation (partial)
- evaluate_fitness: Fitness evaluation function using cross-validation
- binary_pso: Main PSO execution function with adaptive parameter adjustment
'''

import os
import pickle
import time
import warnings
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from PAGEpy.individual_fold_class import IndividualFold
from PAGEpy.models import SimpleNN
from PAGEpy.multiple_folds_class import \
    KFoldData  # TODO: unify folds in a single script

warnings.filterwarnings('ignore')  # Suppress all warnings for cleaner output

# Suppress TensorFlow logging (set to ERROR level to hide INFO and WARNING messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Ensure reproducible results
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow info logs

# Suppress CUDA warnings that are not critical
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA.*')


####################### FITNESS FUNCTION ########################

def reduce_input_features(
        raw_data: IndividualFold, current_genes: list
):
    """
    Subset the input data to include only the selected genes.

    This method filters the feature matrix to include only the genes
    specified in current_genes, effectively implementing feature selection
    for this particular model instance.
    """
    # Find indices of selected genes in the complete gene list
    gene_set_indices = np.where(
        np.isin(raw_data.genes_list, current_genes))[0]

    # Subset training and test data to selected genes only
    x_train = raw_data.x_train[:, gene_set_indices]
    x_test = raw_data.x_test[:, gene_set_indices]

    # Labels remain unchanged
    y_train = raw_data.y_train
    y_test = raw_data.y_test

    return x_train, x_test, y_train, y_test


def evaluate_selected_genes_fitness(
    particle: np.ndarray,
    crossval_folds: KFoldData,
    all_gene_names: list,
    particle_id: int,
    generation_nb: int,
    model_class=None,
    model_params: dict = None,
    training_params: dict = None,
    loud: bool = True
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
        gene for gene, selected in zip(all_gene_names, particle) if selected
    ]
    n_selected_genes = len(selected_gene_names)

    # Set default parameters
    if model_class is None:
        model_class = SimpleNN

    if model_params is None:
        model_params = {
            'n_input_features': n_selected_genes,
        }

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

    n_folds = crossval_folds.n_folds
    fold_aucs = []

    # Perform k-fold cross-validation
    for fold_id in range(n_folds):
        try:
            # Create an IndividualFold object with current fold data
            current_fold = IndividualFold(crossval_folds, fold_id)
            x_train, x_test, y_train, y_test = reduce_input_features(
                current_fold, selected_gene_names
            )

            # Instantiate and train model
            current_model = model_class(**model_params)
            current_model.train(x_train, y_train, **training_params)

            # Evaluate model
            fold_auc = current_model.evaluate(x_test, y_test)
            fold_aucs.append(fold_auc)

            if loud:
                print(f"  Fold {fold_id + 1}/{n_folds}: AUC = {fold_auc:.4f}")

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

######################### PSO ALGORITHM #########################


def moving_average(arr, window_size):
    """
    Calculate moving average for smoothing time series data.

    Args:
        arr (np.ndarray): 2D array where each column represents a time series
        window_size (int): Size of the moving average window

    Returns:
        np.ndarray: Smoothed array with moving averages
    """
    return np.array([
        np.convolve(arr[:, i], np.ones(window_size)/window_size, mode='valid')
        for i in range(arr.shape[1])
    ]).T


class ProgressTracker:  # TODO: refactor this!!
    """
    Track optimization progress using exponential moving average.

    This class maintains a smoothed estimate of optimization progress,
    which helps prevent rapid parameter changes due to noise in fitness values.
    The exponential moving average gives more weight to recent progress while
    still considering historical performance.
    """

    def __init__(self, alpha=0.2):
        """
        Initialize progress tracker.

        Args:
            alpha (float): Smoothing factor (0.1-0.3 recommended)
                          Higher values = more responsive to recent changes
                          Lower values = more stable, less reactive
        """
        self.ema_progress = 0  # Start at neutral (no progress)
        self.alpha = alpha  # Smoothing factor

    def update_progress(self, new_progress):
        """
        Update the exponential moving average of progress.

        Args:
            new_progress (float): Most recent progress measurement

        Returns:
            float: Updated exponential moving average of progress
        """
        self.ema_progress = self.alpha * new_progress + (
            1 - self.alpha) * self.ema_progress
        return self.ema_progress


# NEW: PSO as a class

class BinaryPSO:
    """
    Binary Particle Swarm Optimization for feature selection.

    Attributes:
        POP_SIZE (int): Population size (number of particles)
        W (float): Inertia weight - controls momentum
        C1 (float): Cognitive parameter - attraction to personal best
        C2 (float): Social parameter - attraction to global best
        n_reps (int): Number of fitness evaluation repetitions

        population (np.ndarray): Current population of binary particles
        velocities (np.ndarray): Current velocities of particles
        p_best (np.ndarray): Personal best positions for each particle
        p_best_scores (np.ndarray): Personal best fitness scores
        g_best (np.ndarray): Global best position found by any particle
        g_best_score (float): Global best fitness score
    """

    def __init__(
        self, pop_size, n_features, fitness_function: Callable, w=1, c1=2, c2=2, n_reps=4
    ):
        # PSO Algorithm Parameters
        self.pop_size = pop_size
        self.n_features = n_features
        self.fitness_function = fitness_function
        self.w = w    # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.n_reps = n_reps  # Number of repetitions for fitness evaluation

        # Initialize particle state variables (set during population initialization)
        self.population = None
        self.velocities = None
        self.p_best = None
        self.p_best_scores = None
        self.g_best = None
        self.g_best_score = None

        # Initialize population
        self.initialize_population()

    def initialize_population(self):
        """
        Initialize random population of binary particles and their velocities.
        """
        # Initialize binary positions randomly (0 or 1 for each gene)
        self.population = np.random.randint(
            2, size=(self.pop_size, self.n_features)
        )

        # Initialize small random velocities (will be converted to probabilities)
        self.velocities = np.random.uniform(
            -1, 1, (self.pop_size, self.n_features)
        )

        # Initialize personal bests to current population
        self.p_best = np.copy(self.population)

        # Initialize population best scores to very low values so any fitness
        # will be better
        self.p_best_scores = np.full(self.pop_size, -np.inf)

    def sigmoid(self, x: np.ndarray, alpha: float = 0.8):
        """
        Sigmoid function for converting velocity to probability. The alpha
        parameter controls the steepness of the sigmoid:
        - alpha > 1: Sharper transitions (more binary behavior)
        - alpha < 1: Smoother transitions (better exploration)

        Args:
            x (np.ndarray): Velocity values to convert
            alpha (float): Steepness parameter (default: 0.8)

        Returns:
            np.ndarray: Probabilities in range [0, 1]
        """
        return 1 / (1 + np.exp(-alpha * x))

    def optimize(
        self,
        input_data,  # not sure about the type here
        feature_names: list,
        n_generations: int,
        adaptive_metrics: bool = False,
        print_progress: bool = False
    ):
        # Initialize progress tracking variables
        particle_history = {}  # Store population positions by generation
        fitness_score_history = []  # Store all fitness scores by generation

        if adaptive_metrics:
            # Create tracker with smoothing factor
            progress_tracker = ProgressTracker(alpha=0.2)
            prev_avg_fitness = None
        else:  # to avoid undefined variable error
            progress_tracker = None
            prev_avg_fitness = 0

        crossval_folds = KFoldData(input_data, 5)

        for generation in range(n_generations):
            start_time = time.time()

            # Evaluate fitness for all particles
            fitness_scores = self.evaluate_population(
                crossval_folds=crossval_folds,
                feature_names=feature_names,
                generation_nb=generation+1,
                verbose=print_progress
            )
            avg_fitness = np.mean(fitness_scores)

            # Update personal best
            improved = fitness_scores > self.p_best_scores
            self.p_best[improved] = self.population[improved]
            self.p_best_scores[improved] = fitness_scores[improved]

            # Update global best
            g_best_idx = np.argmax(self.p_best_scores)
            self.g_best = self.p_best[g_best_idx]
            self.g_best_score = self.p_best_scores[g_best_idx]

            # Apply progress-based adjustment for C1 and C2
            if adaptive_metrics and prev_avg_fitness is not None:
                self.progress_based_adjustment(
                    avg_fitness, prev_avg_fitness, progress_tracker
                )
                prev_avg_fitness = avg_fitness

            # Update velocities and positions
            self.update_velocities()
            self.update_positions()

            # Store tracking data
            particle_history[generation] = self.population.copy()
            fitness_score_history.append(
                {f"fitness_p_{i}": score_p_i
                 for i, score_p_i in enumerate(fitness_scores)}
            )

            end_time = time.time()

            print(f"Total time for generation {generation + 1}: "
                  f"{round((end_time - start_time), 2)} seconds")
            print(f"Generation {generation + 1}: Best AUC = {self.g_best_score:.4f}, "
                  f"Avg = {avg_fitness:.4f}")
            print()

        return particle_history, fitness_score_history

    def update_positions(self):
        """
        Update particle positions based on current velocities.
        (binary PSO position update rule)
        """
        prob = self.sigmoid(self.velocities)  # Convert velocity to probability
        self.population = (
            np.random.rand(*self.population.shape) < prob
        ).astype(int)

    def update_velocities(self):
        """
        Update particle velocities using PSO velocity update equation.

        Implements the standard PSO velocity update:
        v = w*v + c1*r1*(p_best - position) + c2*r2*(g_best - position)

        Where:
        - w: inertia weight (maintains current direction)
        - c1: cognitive parameter (attraction to personal best)
        - c2: social parameter (attraction to global best)
        - r1, r2: random factors for stochastic behavior
        """
        # scaling_factor = min(1.0, (gen + 1) / 3)  # Scale up after 3 generations # ???
        r1 = np.random.rand(*self.population.shape)
        r2 = np.random.rand(*self.population.shape)

        self.velocities = (
            self.w * self.velocities +
            self.c1 * r1 * (self.p_best - self.population) +
            self.c2 * r2 * (self.g_best - self.population)
        )

    def evaluate_population(
        self,
        crossval_folds: KFoldData, feature_names: list,
        generation_nb: int, verbose: bool = False
    ):
        """Evaluate fitness for all particles in the population"""

        fitness_scores = np.array([
            np.mean([
                self.fitness_function(
                    particle=particle,
                    crossval_folds=crossval_folds,
                    # qui non va bene che sia ancora all_gene_names ma devo vedere come fare per i params
                    all_gene_names=feature_names,
                    particle_id=particle_id,
                    generation_nb=generation_nb,
                    loud=verbose
                ) for _ in range(self.n_reps)
            ])
            for particle_id, particle in enumerate(self.population)
        ])

        return fitness_scores

    def progress_based_adjustment(
        self, avg_fitness, prev_avg_fitness, progress_tracker, epsilon=1e-6
    ):
        """
        Dynamically adjust PSO parameters C1 and C2 based on optimization progress.

        This adaptive mechanism modifies the balance between exploration (C1) and
        exploitation (C2) based on whether the population is improving or stagnating:
        - If improving: Increase exploitation (C2), decrease exploration (C1)
        - If stagnating: Increase exploration (C1), decrease exploitation (C2)

        Args:
            avg_fitness (float): Current population average fitness
            prev_avg_fitness (float): Previous population average fitness
            C1 (float): Current exploration weight (cognitive parameter)
            C2 (float): Current exploitation weight (social parameter)
            progress_tracker (ProgressTracker): Object tracking smoothed progress
            epsilon (float): Small value to prevent division issues (default: 1e-6)

        Returns:
            tuple: (adjusted_C1, adjusted_C2) - New parameter values

        The function uses exponentially smoothed progress to avoid overreacting
        to single-generation fluctuations in fitness.
        """
        # Calculate raw progress as relative improvement
        raw_progress = (avg_fitness - prev_avg_fitness) / (
            epsilon + abs(prev_avg_fitness))

        # Update smoothed progress estimate
        smoothed_progress = progress_tracker.update_progress(raw_progress)

        print('Current smoothed progress:', round(smoothed_progress, 2))

        # If the progress is too small, do nothing
        if abs(smoothed_progress) < 0.05:
            print("Progress too small, keeping C1 and C2 unchanged.")
            return

        if smoothed_progress > 0:
            # Population is improving - exploit more, explore less
            self.c1 *= (1 - smoothed_progress)  # Decrease exploration
            self.c2 *= (1 + smoothed_progress)  # Increase exploitation
        else:
            # Population is stagnating - explore more, exploit less
            self.c1 *= (1 + abs(smoothed_progress))  # Increase exploration
            self.c2 *= (1 - abs(smoothed_progress))  # Decrease exploitation

        print('Values before normalization:')  # TO DO: remove this print??
        print(self.c1)
        print(self.c2)

        # Constrain parameters to reasonable bounds to prevent instability
        self.c1 = min(max(self.c1, 0.5), 2.5)
        self.c2 = min(max(self.c2, 0.5), 2.5)


# PSO Main Loop

# TODO: add model choice as a parameter!

def run_binary_pso(
    input_data, feature_names: list, pop_size: int, n_generations: int,
    w: float = 1, c1: float = 2, c2: float = 2, n_reps: int = 4,
    verbose: bool = False, adaptive_metrics: bool = False
):
    """
    Run binary Particle Swarm Optimization for feature selection.

    Returns:
        tuple: (best_gene_vector, best_score)
    """

    mixed_precision.set_global_policy('mixed_float16')
    if verbose:
        print(
            f"Using mixed precision policy: {mixed_precision.global_policy().name}"
        )

    # Initialize PSO
    pso = BinaryPSO(
        pop_size=pop_size,
        n_features=len(feature_names),
        fitness_function=evaluate_selected_genes_fitness,
        w=w, c1=c1, c2=c2, n_reps=n_reps)

    # Run PSO to optimize gene selection
    particle_history, fitness_score_history = pso.optimize(
        input_data=input_data,
        feature_names=feature_names,
        n_generations=n_generations,
        adaptive_metrics=adaptive_metrics,
        print_progress=verbose
    )

    # Extract and save selected genes
    selected_genes = [gene for gene, selected in zip(
        feature_names, pso.g_best) if selected == 1]
    print(f"Selected {len(selected_genes)} genes: {selected_genes}")

    with open('pso_selected_genes.pkl', 'wb') as f:
        pickle.dump(selected_genes, f)

    # Save results (history)
    with open("pso_particle_history.pkl", "wb") as f:
        pickle.dump(particle_history, f)
    with open("pso_fitness_scores.pkl", "wb") as f:
        pickle.dump(fitness_score_history, f)

    return pso.g_best, pso.g_best_score
