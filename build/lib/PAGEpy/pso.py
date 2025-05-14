import os
import subprocess
# from multiprocessing import Pool, set_start_method
import warnings
warnings.filterwarnings('ignore')  # Ignores all warnings
# Set the logging level to ERROR (suppress info-level and warning-level messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras import mixed_precision

tf.get_logger().setLevel('ERROR')  # 'WARN' or 'ERROR' will suppress info logs

# Suppress CUDA warnings that are not critical
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA.*')
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from PAGEpy.format_data_class import FormatData
from PAGEpy.multiple_folds_class import MultipleFolds
from PAGEpy.individual_fold_class import IndividualFold
import numpy as np
import random
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
import multiprocessing
from multiprocessing import Manager, Process
import time
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
    
def initiate_fold_wrapper(args):
    """ Wrapper to unpack arguments for initiate_fold """
    return initiate_fold(*args)

def initiate_fold(current_folds, genes_list, fold, fold_name):
    #print(f"Process {fold_name} started.")
    
    genes_list = genes_list.tolist()
    current_fold = IndividualFold(current_folds, fold)
    current_model = PredAnnModel(current_fold, genes_list)
    
    fold_results = current_model.test_auc
    #print(f"Process {fold_name} finished.")
    
    #return fold_name, fold_results
    return fold_results

def evaluate_fitness(individual, current_folds,gene_list, count, gen, loud = True):
    start_time = time.time()

    individual = np.array(individual, dtype=bool)
    gene_list = np.array(gene_list)  
    selected_features = gene_list[individual]

    if len(selected_features) == 0:
        return 0  

    current_member = count + 1
    if loud:
        print(f"Currently training, population member {current_member}, generation {gen}")
    
    fold_names = ['first', 'second', 'third', 'fourth', 'fifth']
    results = {}

    # Sequentially execute folds
    for i in range(5):
        results[fold_names[i]] = initiate_fold_wrapper((current_folds, selected_features, i, fold_names[i]))

    if loud:
        print("All processes completed.")
        
    # Calculate score    
    score = np.mean([results['first'], results['second'], results['third'], results['fourth'], results['fifth']])
    
    # score = round(score,3)
    score = round(float(score), 3)

    if loud:
        print(f"Average final test AUC value: {score}")

    end_time = time.time()
    if loud:
        print(f"Total time: {end_time - start_time} seconds")
    return score


# Sigmoid function for converting velocity to probability
# If alpha > 1, the function becomes sharper (more binary choices).
# If alpha < 1, the function becomes smoother (better feature exploration).

def sigmoid(x, alpha=0.8):
    return 1 / (1 + np.exp(-alpha * x))

# Initialize particles (binary)
def initialize_population(POP_SIZE, FEATURE_COUNT):    
    return np.random.randint(2, size=(POP_SIZE, FEATURE_COUNT))  # Random 0/1

# Initialize velocities
def initialize_velocities(POP_SIZE, FEATURE_COUNT):
    return np.random.uniform(-1, 1, (POP_SIZE, FEATURE_COUNT))  # Small random values

# Binary update rule
def update_positions(population, velocities, POP_SIZE, FEATURE_COUNT):
    prob = sigmoid(velocities)  # Convert velocity to probability
    new_population = (np.random.rand(POP_SIZE, FEATURE_COUNT) < prob).astype(int)  # Flip bits with probability
    return new_population

# Define the training step for only the outcome classifier
def train_step(model, data, outcome_labels):
    with tf.GradientTape() as tape:
        # Forward pass through the outcome classifier
        outcome_predictions = model(data)

        # Compute the biological discriminator loss
        outcome_loss = tf.keras.losses.binary_crossentropy(outcome_labels, outcome_predictions)
        outcome_loss = tf.reduce_mean(outcome_loss)  # Average over the batch

    # Compute gradients for the outcome classifier
    classifier_grads = tape.gradient(outcome_loss, model.trainable_variables)
    
    # Calculate accuracy for the outcome classifier
    predicted_outcome_labels = tf.cast(outcome_predictions > 0.5, tf.float32)  # Threshold at 0.5
    outcome_labels_float = tf.cast(outcome_labels, tf.float32)

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_outcome_labels, outcome_labels_float), tf.float32))

    return outcome_loss, accuracy, classifier_grads

# Set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

class PredAnnModel:
    def __init__(
        self,
        input_data,
        current_genes,
        learning_rate = 0.001,
        dropout_rate=0,
        balance=True,
        l2_reg=0.2,
        batch_size=512,
        num_epochs=50,
        report_frequency=29,
        auc_threshold=0.999,
        clipnorm=0,
        simplify_categories=True,
        multiplier=1,
    ):
        """
        Initializes the PredAnnModel with specified hyperparameters and configuration.

        Parameters:
        - current_genes (list): A non-empty list of genes to be used as model features.
        - learning_rate (float): intial learning rate of the model
        - input_data (RcDataPreparation class object): data for training the model that has been appropriately formattedd.
        - dropout_rate (float): Dropout rate to prevent overfitting (default: 0.3).
        - balance (bool): Whether to balance technology and outcome variables during training (default: True).
        - l2_reg (float): Strength of L2 regularization (default: -0.2).
        - batch_size (int): Batch size for training (default: 16).
        - num_epochs (int): Total number of training epochs (default: 5000).
        - report_frequency (int): Frequency of reporting model metrics (AUC and Accuracy) during training (default: 1).
        - auc_threshold (float): AUC threshold for early stopping (default: 0.9).
        - clipnorm (float): Gradient clipping norm to prevent exploding gradients (default: 2.0).
        - simplify_categories (bool): Whether to simplify categories in the dataset (default: True).
        - multiplier (int): Scales the number of nodes in most network layers (default: 3).
        """

        self.input_data = input_data  # indvidual fold class object for training the model
        self.current_genes = current_genes  # List of genes provided by the user to define model features.
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate  # Dropout rate for regularization.
        self.balance = balance  # Balance technology and outcomes during training.
        self.l2_reg = l2_reg  # Degree of L2 regularization.
        self.batch_size = batch_size  # Batch size for training.
        self.num_epochs = num_epochs  # Total number of training epochs.
        self.report_frequency = report_frequency  # Frequency for collecting metrics during training.
        self.auc_threshold = auc_threshold  # AUC threshold for early stopping.
        self.clipnorm = clipnorm  # Gradient clipping value to prevent exploding gradients.
        self.simplify_categories = simplify_categories  # Whether to reduce data categories (e.g., microarray vs. sequencing).
        self.multiplier = multiplier  # Scales the number of nodes in most layers of the network.
        self.outcome_classifier = None  # ANN model which is instantiated and trained
        self.test_auc = None  # list of metrics for evaluating the model
        self.current_epoch_list = []  # list of epoch numbers for trackking the metrics across models

        # automatically executed functions for establishin and training the model
        self.subset_input_data()
        self.build_outcome_classifier()
        self.train_the_model()
        
    def subset_input_data(self):
        """
        Subsets the data during training.
        """
        gene_set_indices = np.where(np.isin(self.input_data.genes_list, self.current_genes))[0]
        self.x_train = self.input_data.x_train[:, gene_set_indices]
        self.x_test = self.input_data.x_test[:, gene_set_indices]
        self.y_train = self.input_data.y_train
        self.y_test = self.input_data.y_test

    def build_outcome_classifier(self):
        """
        Establishes a faster model.
        """
        self.outcome_classifier = keras.Sequential()
        self.outcome_classifier.add(layers.Input(shape=(len(self.current_genes),)))  # Input shape matches your data

        # Reduced layer sizes for faster training
        self.outcome_classifier.add(layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.ReLU())  # ReLU instead of LeakyReLU for faster computation

        self.outcome_classifier.add(layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.ReLU())  # ReLU instead of LeakyReLU

        self.outcome_classifier.add(layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.ReLU())  # ReLU instead of LeakyReLU

        self.outcome_classifier.add(layers.Dense(32, kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.ReLU())  # ReLU instead of LeakyReLU

        # Output layer for binary classification with sigmoid activation
        self.outcome_classifier.add(layers.Dense(1, activation='sigmoid'))


    def train_the_model(self):
        """
        Trains the model with optimized data handling and computation.
        """
        set_random_seed(seed=42)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        num_samples = math.floor(self.x_train.shape[0])
        num_steps_per_epoch = num_samples // self.batch_size

        # Compile model
        self.outcome_classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Use TensorFlow's AUC metric
        auc_metric = tf.keras.metrics.AUC()
        
        @tf.function
        def train_step(X_batch, y_batch):
            with tf.GradientTape() as tape:
                outcome_predictions = self.outcome_classifier(X_batch, training=True)
                outcome_loss = tf.keras.losses.binary_crossentropy(y_batch, outcome_predictions)
                y_batch = tf.cast(y_batch, tf.float32)
                outcome_predictions = tf.cast(outcome_predictions, tf.float32)
                accuracy = tf.keras.metrics.binary_accuracy(y_batch, outcome_predictions)

            grads = tape.gradient(outcome_loss, self.outcome_classifier.trainable_variables)
            return outcome_loss, accuracy, grads

        for epoch in range(self.num_epochs):
            total_loss, total_accuracy = 0.0, 0.0
            accumulated_grads = [tf.zeros_like(var) for var in self.outcome_classifier.trainable_variables]

            for step in range(num_steps_per_epoch):
                
                batch_indices = []  # Initialize the list
                
                if self.balance:
                    # Get unique class labels
                    unique_classes = np.unique(self.y_train)

                    # Ensure each class is represented equally in the batch
                    for class_label in unique_classes:
                        condition_indices = np.where(self.y_train == class_label)[0]  # Get indices for this class
                        condition_batch_indices = np.random.choice(condition_indices, 
                                                                   size=self.batch_size // len(unique_classes), 
                                                                   replace=True)  # Sample with replacement if needed
                        batch_indices.append(condition_batch_indices)

                    batch_indices = np.concatenate(batch_indices)  # Merge class-specific batches
                else:
                    all_indices = np.arange(len(self.x_train))
                    batch_indices = np.random.choice(all_indices, size=self.batch_size, replace=True)
                
                X_batch = self.x_train[batch_indices]
                y_batch = tf.expand_dims(self.y_train[batch_indices], axis=-1)
                
                outcome_loss, accuracy, grads = train_step(X_batch, y_batch)

                total_loss += outcome_loss.numpy().mean()
                total_accuracy += accuracy.numpy().mean()
                accumulated_grads = [acc_grad + grad for acc_grad, grad in zip(accumulated_grads, grads)]

            averaged_grads = [grad / num_steps_per_epoch for grad in accumulated_grads]
            optimizer.apply_gradients(zip(averaged_grads, self.outcome_classifier.trainable_variables))
            
            if epoch != 0 and epoch % (self.num_epochs-1) == 0:
                
                outcome_predictions = self.outcome_classifier(self.x_test, training=False)
                outcome_labels = tf.expand_dims(self.y_test, axis=-1)
                auc_metric.update_state(outcome_labels, outcome_predictions)
                test_auc = auc_metric.result().numpy()
                
                self.test_auc = test_auc

def moving_average(arr, window_size):
    return np.array([np.convolve(arr[:, i], np.ones(window_size)/window_size, mode='valid') for i in range(arr.shape[1])]).T

class ProgressTracker:
    def __init__(self, alpha=0.2):  # Alpha controls smoothing (0.1-0.3 recommended)
        self.ema_progress = 0  # Start at 0 (neutral)
        self.alpha = alpha  # Smoothing factor

    def update_progress(self, new_progress):
        """Update running exponential moving average of progress."""
        self.ema_progress = self.alpha * new_progress + (1 - self.alpha) * self.ema_progress
        return self.ema_progress
    
def progress_based_adjustment(avg_fitness, prev_avg_fitness, C1, C2, progress_tracker, epsilon=1e-6):
    """
    Adjust C1 and C2 dynamically based on smoothed optimization progress.

    Parameters:
    - avg_fitness: Current population average fitness
    - prev_avg_fitness: Previous population average fitness
    - C1: Current exploration weight
    - C2: Current exploitation weight
    - progress_tracker: Object to track moving average of progress
    - epsilon: Small value to prevent instability in division (default: 1e-6)

    Returns:
    - Adjusted C1 and C2 values
    """
    # Calculate raw progress ratio
    raw_progress = (avg_fitness - prev_avg_fitness) / (epsilon + abs(prev_avg_fitness))

    # Update running progress average
    smoothed_progress = progress_tracker.update_progress(raw_progress)
    
    print('Current smoothed progress:',round(smoothed_progress,2))
    
    # If the progress is too small, do nothing
    if abs(smoothed_progress) < 0.05:
        print("Progress too small, keeping C1 and C2 unchanged.")
        return C1, C2

    if smoothed_progress > 0:  
        # If improving, exploit more (increase C2, reduce C1)
        C1 *= (1 - smoothed_progress)
        C2 *= (1 + smoothed_progress)
    else:  
        # If no improvement, explore more (increase C1, reduce C2)
        C1 *= (1 + abs(smoothed_progress))
        C2 *= (1 - abs(smoothed_progress))

    print('values not normalized:')
    print(C1)
    print(C2)
    
    # Keep values within reasonable bounds
    C1 = min(max(C1, 0.5), 2.5)
    C2 = min(max(C2, 0.5), 2.5)

    return C1, C2
    
# PSO Main Loop

def binary_pso(current_genes, current_data, POP_SIZE, N_GENERATIONS, W = 1, C1 = 2, C2 = 2, reps = 4, frequent_reporting = False, adaptive_metrics = False):
    
    start_time = time.time()
    
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Current mixed precision policy: {mixed_precision.global_policy()}")
    
    # dictionary for saving popluaiton members
    dna_dict = {}  # Empty dictionary
    # making a data frame to keep track of GA progress
    column_names = [f'auc_{i}' for i in range(POP_SIZE)]
    # Initialize an empty DataFrame with columns
    pso_df = pd.DataFrame(columns=column_names)

    
    FEATURE_COUNT = len(current_genes)
        
    population = initialize_population(POP_SIZE, FEATURE_COUNT)  # Random binary solutions
    velocities = initialize_velocities(POP_SIZE, FEATURE_COUNT)  # Random velocities
    
    dna_dict[0] = population
    pickle.dump(dna_dict, open("pso_dict.pkl", "wb"))

    # Personal bests (initially the particles themselves)
    p_best = np.copy(population)
    
    current_folds = MultipleFolds(current_data, 5)
    
    #n = 2
    
    p_best_scores = np.array([
    round(np.mean([evaluate_fitness(ind, current_folds, current_genes, count, 0, loud = frequent_reporting) for _ in range(reps)]), 3)
    for count, ind in enumerate(population)
    ])
    
    # Initialize p_best_scores_history before the loop
    p_best_scores_history = np.zeros((N_GENERATIONS, POP_SIZE))
    p_best_scores_history[0, :] = p_best_scores  # Store initial scores
    
    # save pso results df
    pso_df.loc[len(pso_df)] = p_best_scores
    pso_df.to_pickle("pso_df.pkl")

    # Global best (best solution found by any particle)
    g_best = p_best[np.argmax(p_best_scores)]  # We want the maximum AUC
    g_best_score = max(p_best_scores)
    
    avg_fitness = np.mean(p_best_scores)
    fitness_history = []  # Track progress
    
    # Update the previous best score for the next iteration

    progress_tracker = ProgressTracker(alpha=0.2)  # Create tracker with smoothing factor

    prev_avg_fitness = avg_fitness
    
    end_time = time.time()
    print(f"Total time for generation 1: {round((end_time - start_time),2)} seconds")
    print(f"Generation 1: Best AUC = {g_best_score}, Avg = {avg_fitness}")
    print("\n")

    for gen in range(N_GENERATIONS):
        
        start_time = time.time()
        # Evaluate fitness
        current_folds = MultipleFolds(current_data, 5)
        
        fitness_scores = np.array([
        round(np.mean([evaluate_fitness(ind, current_folds, current_genes, count, gen+1, loud = frequent_reporting) for _ in range(4)]), 3)
        for count, ind in enumerate(population)
        ])
        
        avg_fitness = np.mean(fitness_scores)

        # Store fitness scores history for smoothing
        p_best_scores_history[gen, :] = fitness_scores

        # Update personal bests based on raw fitness scores, not smoothed ones
        improved = fitness_scores > p_best_scores  
        p_best[improved] = population[improved]  # Store best positions
        p_best_scores[improved] = fitness_scores[improved]  # Store best actual scores

        true_best_idx = np.argmax(p_best_scores)  # Use actual best scores
        g_best = p_best[true_best_idx]  # Assign corresponding best position
        g_best_score = p_best_scores[true_best_idx]

        # Apply progress-based adjustment for C1 and C2
        if adaptive_metrics == True:
            C1, C2 = progress_based_adjustment(avg_fitness, prev_avg_fitness, C1, C2, progress_tracker)
                            
        # Update velocities using smoothed p_best
        # scaling_factor = min(1.0, (gen + 1) / 3)  # Scale up after 3 generations
        r1 = np.random.rand(POP_SIZE, FEATURE_COUNT)
        r2 = np.random.rand(POP_SIZE, FEATURE_COUNT)        
        velocities = (
            W * velocities +
            C1 * r1 * (p_best - population) +
            C2 * r2 * (g_best - population))

        # Update positions
        population = update_positions(population, velocities, POP_SIZE, FEATURE_COUNT)

        # Track progress
        avg_fitness = round(np.mean(fitness_scores),3)
        fitness_history.append(avg_fitness)
        
        pso_df.loc[len(pso_df)] = fitness_scores
        pso_df.to_pickle("pso_df.pkl")
        
        dna_dict[gen+1] = population
        pickle.dump(dna_dict, open("pso_dict.pkl", "wb"))
        
        # Update the average score for the next iteration
        prev_best_score = avg_fitness

        end_time = time.time()
        print(f"Total time for generation {gen+2}: {round((end_time - start_time),2)} seconds")
        print(f"Generation {gen+2}: Best AUC = {g_best_score}, Avg = {avg_fitness}")
        print("\n")

    # save the best result for easy accesibley late
    pso_genes = [item for item, m in zip(current_genes, g_best) if m == 1]
    with open('pso_genes_result.pkl', 'wb') as f:
        pickle.dump(pso_genes, f)
    
    return g_best, g_best_score
