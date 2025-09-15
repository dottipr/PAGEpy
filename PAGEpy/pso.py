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

REMARK for future refactoring:
- run_binary_pso is creating a PSO instance and when I call it in the main
  analysis scripts, I'm not using most of the params. The idea is to refactor
  the code mostly from this module until everything is nice (e.g., making 
  run_binary_pso def as simple as possible). Eventually, I'd like to remove it
  as a function and use the PSO class "interface" as much as possible.

Authors: Sean O'Toole and Prisca Dotti
Last modified: 15.09.25
'''

import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from PAGEpy import get_logger
from PAGEpy.dataset_class import GeneExpressionDataset
from PAGEpy.k_folds_class import KFoldData
from PAGEpy.model_adapters import ModelAdapterFactory
from PAGEpy.models import SimpleNN

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

    def __init__(
        self,
        run_id: str,
        feature_names: list,
        pop_size: int,
        model_class,
        model_params: dict,
        w: float = 1,
        c1: float = 2,
        c2: float = 2,
        n_reps: int = 4,
        checkpoint_dir: str = "pso_checkpoints"
    ):
        """
        Initialize Binary PSO with model configuration.

        Args:
            run_id: Unique identifier
            pop_size: Population size  
            XXXXX n_features: Number of features
            model_class: Model to use (RandomForestClassifier, SimpleNN, etc.)
            model_params: Parameters for model initialization
            fitness_function: Custom fitness function (uses evaluate_particle_fitness if None)
            w, c1, c2: PSO parameters (inertia weight, cognitive param, social param)
            n_reps: Number of fitness evaluation repetitions
            checkpoint_dir: Checkpoint directory
        """

        # PSO Algorithm Parameters
        self.run_id = run_id
        self.pop_size = pop_size
        self.feature_names = feature_names
        self.n_features = len(self.feature_names)
        self.w, self.c1, self.c2 = w, c1, c2
        self.n_reps = n_reps

        # Model configuration
        self.model_class = model_class
        self.model_params = model_params

        # Checkpointing setup
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # PSO state
        self.current_generation = 0
        self.population: np.ndarray
        self.velocities: np.ndarray
        self.p_best: np.ndarray
        self.p_best_scores: np.ndarray
        self.g_best: np.ndarray
        self.g_best_score: float

        # Initialize population or load from checkpoint
        if not self._load_checkpoint():
            self._initialize_population()

    def _initialize_population(self) -> None:
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

    def _sigmoid(self, x: np.ndarray, alpha: float = 0.8) -> np.ndarray:
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

    def _update_positions(self) -> None:
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

        # Convert velocity to probability
        prob = self._sigmoid(self.velocities)
        self.population = (
            np.random.rand(*self.population.shape) < prob
        ).astype(int)

    def _update_velocities(self) -> None:
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

    def optimize(
        self,
        input_data: GeneExpressionDataset,
        n_generations: int,
        run_params: dict,
        adaptive_metrics: bool = False,
        verbose: bool = False
    ) -> Tuple[List[str], Dict[int, np.ndarray], List[Dict[str, float]]]:
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

        ################ Try to restore history from checkpoint ################

        if self.current_generation > 0:
            checkpoint_path = self._get_checkpoint_path()
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    particle_history = checkpoint_data.get(
                        'particle_history', {})
                    fitness_score_history = checkpoint_data.get(
                        'fitness_score_history', [])
                    logger.info(
                        "Restored history with %d generations", len(particle_history))
            except Exception as e:
                logger.warning("Could not restore history: %s", str(e))

        ######################### Run PSO Optimization #########################

        # Start from current generation (0 if fresh start, or resumed generation)
        for generation in range(self.current_generation, n_generations):
            logger.info("="*60)
            logger.info("Generation %d started", generation + 1)
            start_time = time.time()

            # Evaluate fitness for all particles
            fitness_scores = np.array(self._evaluate_population(
                input_data=input_data,
                run_params=run_params,
                verbose=verbose,
            ))
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
                self._progress_based_adjustment(
                    avg_fitness, prev_avg_fitness, progress_tracker
                )
                prev_avg_fitness = avg_fitness

            # Update velocities and positions for next generation
            if generation < n_generations - 1:  # Don't update on last generation
                self._update_velocities()
                self._update_positions()

            # Store tracking data
            particle_history[generation] = self.population.copy()
            fitness_score_history.append(
                {f"fitness_p_{i}": score_p_i
                 for i, score_p_i in enumerate(fitness_scores)}
            )

            # Update current generation and save checkpoint
            self.current_generation = generation + 1
            self._save_checkpoint(
                generation=self.current_generation,
                particle_history=particle_history,
                fitness_score_history=fitness_score_history
            )

            end_time = time.time()

            logger.info(
                "Generation %d summary: Best AUC: %.4f | Average AUC: %.4f | Duration: %.2fs",
                generation + 1, self.g_best_score, avg_fitness, end_time - start_time
            )

        ################### Algorithm completed successfully ###################

        # Delete checkpoint
        self._delete_checkpoint()

        # Extract selected genes
        selected_genes = [gene for gene, selected in zip(
            self.feature_names, self.g_best) if selected]

        logger.info("Selected %d features", len(selected_genes))
        logger.debug("Selected features: %s", ", ".join(
            selected_genes) if selected_genes else "None")

        return selected_genes, particle_history, fitness_score_history

    def _evaluate_population(
            self, input_data: GeneExpressionDataset, run_params: dict, verbose: bool = False,
    ) -> list:
        """Evaluate fitness for all particles in the population using cross-validation."""
        if self.population is None:
            raise ValueError(
                "self.population must be initialized before evaluating fitness.")

        fitness_scores = []

        for particle_id, particle in enumerate(self.population):
            particle_scores = []

            for rep in range(self.n_reps):
                start_time = time.time()

                # Convert individual to boolean array for gene selection
                selected_features = [
                    feat_name for feat_name, selected in zip(
                        self.feature_names, particle) if selected
                ]
                n_selected = len(selected_features)

                if n_selected == 0:
                    logger.warning(
                        "Particle %d, repetition %d: No features selected. Fitness set to 0.",
                        particle_id + 1, rep + 1
                    )
                    particle_scores.append(0.0)

                # Create model adapter to have train_and_score method
                model_wrapper = ModelAdapterFactory.create_adapter(
                    model_class=self.model_class,
                    model_params=self.model_params,
                    n_features=n_selected
                )

                # Create cross-validation folds
                kfolds = KFoldData(input_data, 5)
                fold_scores = []

                # Perform k-fold cross-validation
                for fold_id, fold in enumerate(kfolds):
                    try:
                        x_train, x_test, y_train, y_test = fold.reduce_input_features(
                            selected_features)

                        # Train model and get average AUC score
                        avg_score = model_wrapper.train_and_score(
                            x_train, y_train, x_test, y_test, run_params
                        )

                        fold_scores.append(avg_score)

                        if verbose:
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
                        # Assign zero fitness for failed folds
                        fold_scores.append(0.0)

                avg_score = np.mean(fold_scores)
                particle_scores.append(avg_score)
                end_time = time.time()

                if verbose:
                    logger.info(
                        "Particle %d: Genes=%d | Mean AUC=%.3f | Time=%.2fs",
                        particle_id + 1, n_selected, avg_score, end_time - start_time
                    )
            fitness_scores.append(np.mean(particle_scores))

        return fitness_scores

    def _progress_based_adjustment(
        self, avg_fitness, prev_avg_fitness, progress_tracker, epsilon=1e-6
    ) -> None:
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

    def _get_checkpoint_path(self) -> str:
        """Get the checkpoint file path for this run."""
        return os.path.join(self.checkpoint_dir, f"checkpoint_{self.run_id}.pkl")

    def _save_checkpoint(
        self, generation: int, particle_history: Dict, fitness_score_history: list
    ) -> None:
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

        checkpoint_path = self._get_checkpoint_path()
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        logger.info(
            "Checkpoint saved at generation %d: %s", generation, checkpoint_path)

    def _load_checkpoint(self) -> bool:
        """
        Load PSO state from checkpoint file if it exists.

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        checkpoint_path = self._get_checkpoint_path()

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

    def _delete_checkpoint(self) -> None:
        """Delete the checkpoint file for this run."""
        checkpoint_path = self._get_checkpoint_path()
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info("Checkpoint deleted: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to delete checkpoint: %s", e)


# PSO Main Loop

def run_binary_pso(
    run_id: str,
    input_data: GeneExpressionDataset,
    feature_names: list,
    pop_size: int,
    n_generations: int,
    model_class=SimpleNN,
    model_params={  # specific to the considered model_class (e.g., sklearn methods -> empty (??))
        'learning_rate': 0.001,
        'l2_reg': 0.2,
        'balance_classes': True,
        'report_frequency': 10,
        'auc_threshold': 0.999,
    },
    # TODO: maybe run_params is used only when model_params is a custom NN????
    # in tal caso: aggiungere ai model_params
    run_params={'n_epochs': 50, 'batch_size': 512, },
    w: float = 1,
    c1: float = 2,
    c2: float = 2,
    n_reps: int = 4,
    verbose: bool = False,
    adaptive_metrics: bool = False,
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
    ################# Initialise checkpointing and PSO object ##################

    # Set up checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            os.path.dirname(output_prefix), "checkpoints")

    # Initialize PSO
    pso = BinaryPSO(
        run_id=run_id,
        feature_names=feature_names,
        pop_size=pop_size,
        model_class=model_class,
        model_params=model_params,
        w=w, c1=c1, c2=c2, n_reps=n_reps,
        checkpoint_dir=checkpoint_dir)

    logger.info(
        "Starting PSO optimization with run_id '%s' and model class '%s'",
        pso.run_id, str(model_class.__name__))
    if pso.current_generation > 0:
        logger.info("Resuming from generation %d", pso.current_generation)

    ########################## PSO optimization loop ###########################

    # Run PSO to optimize gene selection
    selected_genes, particle_history, fitness_score_history = pso.optimize(
        input_data=input_data,
        n_generations=n_generations,
        run_params=run_params,
        adaptive_metrics=adaptive_metrics,
        verbose=verbose
    )

    ###################### Save, plot, and return results ######################

   # Save selected genes
    os.makedirs(output_prefix, exist_ok=True)
    with open(os.path.join(output_prefix, 'pso_selected_genes.pkl'), 'wb') as f:
        pickle.dump(selected_genes, f)

    # Also save selected genes as a .txt file (one gene per line)
    with open(
        os.path.join(output_prefix, 'pso_selected_genes.txt'), 'w', encoding='utf-8'
    ) as txt_f:
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
