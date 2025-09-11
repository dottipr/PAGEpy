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
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from PAGEpy import get_logger
from PAGEpy.fitness_functions import evaluate_particle_fitness
from PAGEpy.k_folds_class import KFoldData

logger = get_logger(__name__)

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


class BinaryPSO:
    """
    Binary Particle Swarm Optimization for feature selection.

    Attributes:
        run_id (str): Unique identifier for this run
        pop_size (int): Population size (number of particles)
        w (float): Inertia weight - controls momentum
        c1 (float): Cognitive parameter - attraction to personal best
        c2 (float): Social parameter - attraction to global best
        n_reps (int): Number of fitness evaluation repetitions

        population (np.ndarray): Current population of binary particles
        velocities (np.ndarray): Current velocities of particles
        p_best (np.ndarray): Personal best positions for each particle
        p_best_scores (np.ndarray): Personal best fitness scores
        g_best (np.ndarray): Global best position found by any particle
        g_best_score (float): Global best fitness score

        # Checkpointing attributes
        checkpoint_dir (str): Directory to store checkpoints
        current_generation (int): Current generation number
    """

    # TODO: in realtà potrei magari instanziare il modello al di fuori
    #       dell'evaluate_population ?? (così invece di passare model class +
    #       model params, basta passare il modello già creato, anché perché non
    #       è il compito di una PSO class creare un DL model!)

    def __init__(
        self,
        run_id: str,
        pop_size: int,
        n_features: int,
        # TODO: devo davvero passarla? non dovrebbe essercene una generica?
        fitness_function: Callable,
        model_class,
        hyperparams: Optional[dict] = None,
        training_params: Optional[dict] = None,
        w: float = 1,
        c1: float = 2,
        c2: float = 2,
        n_reps: int = 4,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize Binary PSO with model configuration.

        Args:
            run_id: Unique identifier
            pop_size: Population size  
            n_features: Number of features
            model_class: Model to use (RandomForestClassifier, SimpleNN, etc.)
            hyperparams: Parameters for model initialization
            training_params: Parameters for model training/fitting
            fitness_function: Custom fitness function (uses evaluate_particle_fitness if None)
            w, c1, c2: PSO parameters
            n_reps: Number of fitness evaluation repetitions
            checkpoint_dir: Checkpoint directory
        """

        # PSO Algorithm Parameters
        self.run_id = run_id
        self.pop_size = pop_size
        self.n_features = n_features
        self.fitness_function = fitness_function
        self.w = w    # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.n_reps = n_reps  # Number of repetitions for fitness evaluation

        # Model configuration
        self.model_class = model_class
        self.hyperparams = hyperparams or {}
        self.training_params = training_params or {}

        # Fitness function TODO: fitness_function arg can be removed
        self.fitness_function = fitness_function or evaluate_particle_fitness

        # Checkpointing setup
        self.checkpoint_dir = checkpoint_dir or "pso_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.current_generation = 0

        # PSO state
        self.population: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.p_best: Optional[np.ndarray] = None
        self.p_best_scores: Optional[np.ndarray] = None
        self.g_best: Optional[np.ndarray] = None
        self.g_best_score: Optional[float] = None

        # Initialize population or load from checkpoint
        if not self.load_checkpoint():
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

    def update_positions(self):
        """
        Update particle positions based on current velocities.
        (binary PSO position update rule)
        """
        if self.velocities is None:
            raise ValueError(
                "self.velocities must be initialized before updating velocities.")
        if self.population is None:
            raise ValueError(
                "self.population must be initialized before updating positions.")

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
        if self.population is None:
            raise ValueError(
                "self.population must be initialized before updating population.")
        if self.velocities is None:
            raise ValueError(
                "self.velocities must be initialized before updating velocities.")

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
        kfolds: KFoldData, feature_names: list,
        generation_nb: int, verbose: bool = False,
    ):
        """Evaluate fitness for all particles in the population"""

        if self.population is None:
            raise ValueError(
                "self.population must be initialized before evaluating fitness.")

        fitness_scores = np.array([
            np.mean([
                self.fitness_function(
                    particle=particle,
                    kfolds=kfolds,
                    feature_names=feature_names,
                    particle_id=particle_id,
                    generation_nb=generation_nb,
                    loud=verbose,
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

        logger.info("Current smoothed progress: %.2f", smoothed_progress)

        # If the progress is too small, do nothing
        if abs(smoothed_progress) < 0.05:
            logger.info("Progress too small, keeping C1 and C2 unchanged.")
            return

        if smoothed_progress > 0:
            # Population is improving - exploit more, explore less
            self.c1 *= (1 - smoothed_progress)  # Decrease exploration
            self.c2 *= (1 + smoothed_progress)  # Increase exploitation
        else:
            # Population is stagnating - explore more, exploit less
            self.c1 *= (1 + abs(smoothed_progress))  # Increase exploration
            self.c2 *= (1 - abs(smoothed_progress))  # Decrease exploitation

        logger.info(
            "Values before normalization: c1=%.4f, c2=%.4f",
            self.c1, self.c2
        )

        # Constrain parameters to reasonable bounds to prevent instability
        self.c1 = min(max(self.c1, 0.5), 2.5)
        self.c2 = min(max(self.c2, 0.5), 2.5)

    def get_checkpoint_path(self) -> str:
        """Get the checkpoint file path for this run."""
        return os.path.join(self.checkpoint_dir, f"checkpoint_{self.run_id}.pkl")

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

        # Try to restore history from checkpoint
        if self.current_generation > 0:
            checkpoint_path = self.get_checkpoint_path()
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    particle_history = checkpoint_data.get(
                        'particle_history', {})
                    fitness_score_history = checkpoint_data.get(
                        'fitness_score_history', [])
                    logger.info(
                        f"Restored history with {len(particle_history)} generations")
            except Exception as e:
                logger.warning(f"Could not restore history: {e}")

        if adaptive_metrics:
            # Create tracker with smoothing factor
            progress_tracker = ProgressTracker(alpha=0.2)
            prev_avg_fitness = None
        else:  # to avoid undefined variable error
            progress_tracker = None
            prev_avg_fitness = 0

        crossval_folds = KFoldData(input_data, 5)

        # Start from current generation (0 if fresh start, or resumed generation)
        for generation in range(self.current_generation, n_generations):
            logger.info("="*60)
            logger.info("Generation %d started", generation + 1)
            start_time = time.time()

            # Evaluate fitness for all particles
            fitness_scores = self.evaluate_population(
                kfolds=crossval_folds,
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

            # Update velocities and positions for next generation
            if generation < n_generations - 1:  # Don't update on last generation
                self.update_velocities()
                self.update_positions()

            # Store tracking data
            particle_history[generation] = self.population.copy()
            fitness_score_history.append(
                {f"fitness_p_{i}": score_p_i
                 for i, score_p_i in enumerate(fitness_scores)}
            )

            # Update current generation and save checkpoint
            self.current_generation = generation + 1
            self.save_checkpoint(
                generation=self.current_generation,
                particle_history=particle_history,
                fitness_score_history=fitness_score_history
            )

            end_time = time.time()

            logger.info(
                "Generation %d summary: Best AUC: %.4f | Average AUC: %.4f | Duration: %.2fs",
                generation + 1, self.g_best_score, avg_fitness, end_time - start_time
            )

        # Algorithm completed successfully - delete checkpoint
        self.delete_checkpoint()

        return particle_history, fitness_score_history

    def save_checkpoint(
        self, generation: int, particle_history: Dict, fitness_score_history: list
    ):
        """
        Save current PSO state to checkpoint file.

        Args:
            generation (int): Current generation number
            particle_history (Dict): History of particle positions
            fitness_score_history (list): History of fitness scores
        """
        checkpoint_data = {
            # PSO state
            'population': self.population,
            'velocities': self.velocities,
            'p_best': self.p_best,
            'p_best_scores': self.p_best_scores,
            'g_best': self.g_best,
            'g_best_score': self.g_best_score,

            # Parameters (in case they were modified by adaptive metrics)
            'w': self.w,
            'c1': self.c1,
            'c2': self.c2,

            # Progress tracking
            'current_generation': generation,
            'particle_history': particle_history,
            'fitness_score_history': fitness_score_history,

            # Metadata
            'pop_size': self.pop_size,
            'n_features': self.n_features,
            'n_reps': self.n_reps,
            'run_id': self.run_id,
        }

        checkpoint_path = self.get_checkpoint_path()
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        logger.info(
            "Checkpoint saved at generation %d: %s", generation, checkpoint_path)

    def load_checkpoint(self) -> bool:
        """
        Load PSO state from checkpoint file if it exists.

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        checkpoint_path = self.get_checkpoint_path()

        if not os.path.exists(checkpoint_path):
            logger.info(
                "No checkpoint found for run_id '%s'. Starting fresh.", self.run_id)
            return False

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Validate compatibility
            if (checkpoint_data['pop_size'] != self.pop_size or
                    checkpoint_data['n_features'] != self.n_features):
                logger.warning(
                    "Checkpoint parameters don't match current configuration. Starting fresh.")
                return False

            # Restore PSO state
            self.population = checkpoint_data['population']
            self.velocities = checkpoint_data['velocities']
            self.p_best = checkpoint_data['p_best']
            self.p_best_scores = checkpoint_data['p_best_scores']
            self.g_best = checkpoint_data['g_best']
            self.g_best_score = checkpoint_data['g_best_score']

            # Restore parameters
            self.w = checkpoint_data['w']
            self.c1 = checkpoint_data['c1']
            self.c2 = checkpoint_data['c2']

            # Restore progress
            self.current_generation = checkpoint_data['current_generation']

            logger.info(
                "Checkpoint loaded successfully. Resuming from generation %d",
                self.current_generation)
            logger.info("Previous best score: %.4f", self.g_best_score)

            return True

        except Exception as e:
            logger.error("Failed to load checkpoint: %s. Starting fresh.", e)
            return False

    def delete_checkpoint(self):
        """Delete the checkpoint file for this run."""
        checkpoint_path = self.get_checkpoint_path()
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info("Checkpoint deleted: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to delete checkpoint: %s", e)


# PSO Main Loop

# TODO: add model choice as a parameter!


def run_binary_pso(
    run_id: str, input_data, feature_names: list, pop_size: int, n_generations: int,
    w: float = 1, c1: float = 2, c2: float = 2, n_reps: int = 4,
    verbose: bool = False, adaptive_metrics: bool = False,
    output_prefix: str = "pso_results",
    checkpoint_dir: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Run binary Particle Swarm Optimization for feature selection with checkpointing.

    Args:
        run_id (str): Unique run identifier
        input_data: Input data for fitness evaluation
        feature_names (list): List of feature names
        pop_size (int): Population size
        n_generations (int): Number of generations to run
        w (float): Inertia weight (default: 1)
        c1 (float): Cognitive parameter (default: 2)
        c2 (float): Social parameter (default: 2)
        n_reps (int): Number of fitness evaluation repetitions (default: 4)
        verbose (bool): Whether to print progress (default: False)
        adaptive_metrics (bool): Whether to use adaptive parameter adjustment (default: False)
        output_prefix (str): Prefix for output files (default: "pso_results")
        checkpoint_dir (str, optional): Directory for checkpoints. If None, uses "pso_checkpoints"

    Returns:
        tuple: (best_gene_vector, best_score)
    """
    # Set up checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            os.path.dirname(output_prefix), "checkpoints")

    # Initialize PSO
    pso = BinaryPSO(
        run_id=run_id,
        pop_size=pop_size,
        n_features=len(feature_names),
        fitness_function=evaluate_particle_fitness,
        w=w, c1=c1, c2=c2, n_reps=n_reps,
        checkpoint_dir=checkpoint_dir)

    logger.info("Starting PSO optimization with run_id: %s", pso.run_id)
    if pso.current_generation > 0:
        logger.info("Resuming from generation %d", pso.current_generation)

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
    logger.info("Selected %d genes (top features)", len(selected_genes))
    logger.debug("Selected genes: %s", ", ".join(
        selected_genes) if selected_genes else "None")

    # Ensure output directory exists
    os.makedirs(output_prefix, exist_ok=True)

    with open(os.path.join(output_prefix, 'pso_selected_genes.pkl'), 'wb') as f:
        pickle.dump(selected_genes, f)

    # Also save selected genes as a .txt file (one gene per line)
    with open(os.path.join(output_prefix, 'pso_selected_genes.txt'), 'w', encoding='utf-8') as txt_f:
        for gene in selected_genes:
            txt_f.write(f"{gene}\n")

    # Save results (history)
    with open(os.path.join(output_prefix, 'pso_particle_history.pkl'), "wb") as f:
        pickle.dump(particle_history, f)
    with open(os.path.join(output_prefix, 'pso_fitness_scores.pkl'), "wb") as f:
        pickle.dump(fitness_score_history, f)

    # Save final results with run_id for reference
    final_results = {
        'run_id': pso.run_id,
        'best_solution': pso.g_best,
        'best_score': pso.g_best_score,
        'selected_genes': selected_genes,
        'final_parameters': {'w': pso.w, 'c1': pso.c1, 'c2': pso.c2},
        'completed_at': datetime.now().isoformat()
    }

    with open(os.path.join(output_prefix, 'pso_final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    logger.info(
        "PSO optimization completed successfully. Run ID: %s", pso.run_id)

    return pso.g_best, pso.g_best_score
