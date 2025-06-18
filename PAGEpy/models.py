import math

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, mixed_precision

# from PAGEpy.config import Config  # TO DO: maybe remove, not used in the class

# should increase model efficiency
# not cpu compatible, commented out for now


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across numpy and TensorFlow.

    This ensures that results can be reproduced across different runs
    by controlling the random number generation in both libraries.

    Args:
        seed (int): Random seed value (default: 42)

    # TO DO: random seeds are being set a bit allover the place -> fix this
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --> TO DO: devo trovare il modo di non avere due definizioni del train_step
# Define the training step for only the outcome classifier


def train_step(model, data, outcome_labels):
    """
    Custom training step for the neural network classifier.

    This function performs a single forward and backward pass through the network,
    computing loss, accuracy, and gradients for the outcome classification task.

    Args:
        model: TensorFlow/Keras model to train
        data: Input features (gene expression data)
        outcome_labels: Binary target labels for classification

    Returns:
        tuple: (outcome_loss, accuracy, classifier_grads)
            - outcome_loss: Binary cross-entropy loss value
            - accuracy: Classification accuracy for this batch
            - classifier_grads: Gradients for model parameters
    """
    with tf.GradientTape() as tape:
        # Forward pass: get model predictions
        outcome_predictions = model(data)

        # Compute binary cross-entropy loss
        outcome_loss = tf.keras.losses.binary_crossentropy(
            outcome_labels, outcome_predictions)
        outcome_loss = tf.reduce_mean(outcome_loss)  # Average loss over batch

    # Compute gradients for backpropagation
    classifier_grads = tape.gradient(outcome_loss, model.trainable_variables)

    # Calculate binary classification accuracy
    predicted_outcome_labels = tf.cast(
        outcome_predictions > 0.5, tf.float32)  # Threshold at 0.5
    outcome_labels_float = tf.cast(outcome_labels, tf.float32)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted_outcome_labels, outcome_labels_float), tf.float32))

    return outcome_loss, accuracy, classifier_grads

# TO DO: qui c'è ancora una definizione di train_step!!!


def train_step(model, data, outcome_labels):
    with tf.GradientTape() as tape:
        # Forward pass through the outcome classifier
        outcome_predictions = model(data)

        # Compute the biological discriminator loss
        outcome_loss = tf.keras.losses.binary_crossentropy(
            outcome_labels, outcome_predictions)
        outcome_loss = tf.reduce_mean(outcome_loss)  # Average over the batch

    # Compute gradients for the outcome classifier
    classifier_grads = tape.gradient(outcome_loss, model.trainable_variables)

    # Calculate accuracy for the outcome classifier
    predicted_outcome_labels = tf.cast(
        outcome_predictions > 0.5, tf.float32)  # Threshold at 0.5
    outcome_labels_float = tf.cast(outcome_labels, tf.float32)

    # Calculate accuracy
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted_outcome_labels, outcome_labels_float), tf.float32))

    return outcome_loss, accuracy, classifier_grads


class SimpleNN:
    """
    Simple Artificial Neural Network Model for binary classification
    of genomic data.

    This class encapsulates a multi-layer neural network designed to
    classify samples based on selected gene expression features. It
    automatically handles data subsetting, model architecture creation,
    training, and evaluation.

    Architecture:
        - Input layer (variable size based on selected genes)
        - Dense layer: 256 units with ReLU activation and L2 regularization
        - Dense layer: 128 units with ReLU activation and L2 regularization
        - Dense layer: 64 units with ReLU activation and L2 regularization
        - Dense layer: 32 units with ReLU activation
        - Output layer: 1 unit with sigmoid activation (binary classification)

    The model uses He normal initialization, Adam optimizer, and binary
    cross-entropy loss.
    """

    def __init__(
        self,
        n_input_features,
        learning_rate=0.001,
        dropout_rate=0,
        l2_reg=0.2,
        auc_threshold=0.999,
        balance=True,
        report_frequency=29,
        clipnorm=0,
        simplify_categories=True,
        multiplier=1,
    ):
        """
        Initializes the SimpleNNModel with specified gene features.

        Args: TO DO
            input_data (IndividualFold): Preprocessed data object containing train/test splits
            current_genes (list): List of gene names to use as model features
            balance (bool): Whether to balance classes in training batches (default: True)
            report_frequency (int): How often to report metrics during training (default: 1 ???)
            clipnorm (float): Gradient clipping norm (default: 2.0 ??? / 0, no clipping)
            simplify_categories (bool): Whether to simplify data categories (default: True)
            multiplier (int): Scales the number of nodes in most layers of the network (default: 1, not currently used/ default: 3 ???)

        Attributes:
            test_auc (float): Final AUC score on test set after training
        """

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.balance = balance
        self.l2_reg = l2_reg  # L2 regularization strength
        self.report_frequency = report_frequency
        self.auc_threshold = auc_threshold  # AUC threshold for early stopping
        self.clipnorm = clipnorm
        self.simplify_categories = simplify_categories
        self.multiplier = multiplier
        self.n_genes = n_input_features

        # Initialise model attributes
        self.model = None
        self.optimizer = None  # Optimizer for training
        self.test_auc = None  # list of metrics for evaluating the model

        self.build_model()  # Create the neural network architecture

        # Initialize AUC metric for evaluation
        self.auc_metric = tf.keras.metrics.AUC()

    def build_model(self):
        """
        Build the neural network architecture.

        Creates a feedforward neural network with multiple dense layers,
        ReLU activations, L2 regularization, and He normal initialization.
        The architecture is designed for binary classification of genomic data.
        """
        self.model = tf.keras.Sequential([
            # Input layer - size matches number of selected genes
            tf.keras.layers.Input(shape=(self.n_genes,)),
            tf.keras.layers.Dense(
                256,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                kernel_initializer='he_normal',  # Good for ReLU activations
                activation='relu'
            ),
            tf.keras.layers.Dense(
                128,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                kernel_initializer='he_normal',
                activation='relu'
            ),
            tf.keras.layers.Dense(
                64,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                kernel_initializer='he_normal',
                activation='relu'
            ),
            # no L2 reg on smaller layer
            tf.keras.layers.Dense(
                32,
                kernel_initializer='he_normal',
                activation='relu'
            ),
            # Output layer: single unit with sigmoid for binary classification
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Configure optimizer and compile model
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',  # Standard loss for binary classification
            metrics=['accuracy']
        )

    def evaluate(self, x_test, y_test):
        outcome_predictions = self.model(x_test, training=False)
        outcome_labels = tf.expand_dims(y_test, axis=-1)
        self.auc_metric.update_state(outcome_labels, outcome_predictions)
        self.test_auc = self.auc_metric.result().numpy()

        return self.test_auc

    @tf.function  # Compile for faster execution
    def train_step(self, x_batch, y_batch):
        """Inner training step function with automatic differentiation."""
        with tf.GradientTape() as tape:
            # Forward pass
            outcome_predictions = self.model(x_batch, training=True)
            outcome_loss = tf.keras.losses.binary_crossentropy(
                y_batch, outcome_predictions
            )

            # Compute gradients
            y_batch = tf.cast(y_batch, tf.float32)
            outcome_predictions = tf.cast(outcome_predictions, tf.float32)
            accuracy = tf.keras.metrics.binary_accuracy(
                y_batch, outcome_predictions
            )

        # Compute gradients
        grads = tape.gradient(outcome_loss, self.model.trainable_variables)
        return outcome_loss, accuracy, grads

    def train(
        self, x_train, y_train,
        n_epochs: int = 100, batch_size: int = 32, seed=42
    ):
        """
        Train the neural network model with optimized batch handling.

        This method implements a custom training loop with:
        - Optional class balancing in batch construction
        - Manual gradient accumulation and application
        - Efficient TensorFlow operations using @tf.function decoration
        - Final model evaluation on test set

        The training process accumulates gradients over multiple batches
        per epoch and applies them once per epoch for computational efficiency.
        """
        set_random_seed(seed=seed)  # Ensure reproducible results

        # Calculate training parameters
        # TO DO: Why math.floor??
        n_samples = math.floor(x_train.shape[0])
        n_steps_per_epoch = n_samples // batch_size

        # Main training loop
        for epoch in range(n_epochs):
            total_loss, total_accuracy = 0.0, 0.0
            # Initialize gradient accumulation
            accumulated_grads = [tf.zeros_like(
                var) for var in self.model.trainable_variables]

            # Process all batches in this epoch
            for _ in range(n_steps_per_epoch):
                batch_indices = []

                if self.balance:
                    # Create balanced batches with equal representation from each class
                    unique_classes = np.unique(y_train)

                    for class_label in unique_classes:
                        # Get indices for this class
                        condition_indices = np.where(
                            y_train == class_label
                        )[0]
                        # Sample equal number from each class
                        condition_batch_indices = np.random.choice(
                            condition_indices,
                            size=batch_size // len(unique_classes),
                            replace=True  # Allow replacement if class is small
                        )
                        batch_indices.append(condition_batch_indices)
                    # Merge class-specific batches
                    batch_indices = np.concatenate(batch_indices)

                else:
                    # Random sampling without class balancing
                    all_indices = np.arange(len(x_train))
                    batch_indices = np.random.choice(
                        all_indices, size=batch_size, replace=True
                    )

                #  Create batch data
                x_batch = x_train[batch_indices]
                y_batch = tf.expand_dims(y_train[batch_indices], axis=-1)

                # Execute training step
                outcome_loss, accuracy, grads = self.train_step(
                    x_batch, y_batch)

                # Accumulate metrics and gradients
                total_loss += outcome_loss.numpy().mean()
                total_accuracy += accuracy.numpy().mean()
                accumulated_grads = [
                    acc_grad + grad
                    for acc_grad, grad in zip(accumulated_grads, grads)
                ]

            # Apply accumulated gradients (once per epoch)
            averaged_grads = [
                grad / n_steps_per_epoch for grad in accumulated_grads
            ]
            self.optimizer.apply_gradients(
                zip(averaged_grads, self.model.trainable_variables))


def adjust_learning_rate_by_auc(epoch, model, x_test, y_test_outcome, lr_dict, auc_thresholds, test_auc):
    # Get model's current learning rate
    # current_lr = tf.keras.backend.get_value(model.optimizer.inner_optimizer.learning_rate) #tf.keras.backend.get_value(model.optimizer.lr)
    current_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)

    # Adjust learning rate based on AUC thresholds
    new_lr = current_lr
    for threshold in auc_thresholds:
        if test_auc >= threshold:
            new_lr = lr_dict[threshold]

    # Set new learning rate
    if new_lr != current_lr:
        # tf.keras.backend.set_value(model.optimizer.lr, new_lr)
        # tf.keras.backend.set_value(model.optimizer.inner_optimizer.learning_rate, new_lr)
        # print(f"Epoch {epoch + 1}: Adjusting learning rate to {new_lr:.6f}")
        # tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        model.optimizer.learning_rate = new_lr

    return test_auc


# TODO: devo ancora fare il refactoring di questa classe!!
class PredAnnModel:
    def __init__(
        self,
        input_data,
        current_genes,
        learning_rate=0.01,
        dropout_rate=0.3,
        balance=True,
        l2_reg=0.2,
        batch_size=64,
        n_epochs=5000,
        report_frequency=1,
        auc_threshold=0.95,
        clipnorm=2.0,
        simplify_categories=True,
        holdout_size=0.5,
        multiplier=3,
        auc_thresholds=[0.6, 0.7, 0.8, 0.85, 0.88, 0.89, 0.90, 0.91, 0.92],
        lr_dict={
            # 0.6:  0.005,
            # 0.7:  0.001,
            # 0.8:  0.0005,
            # 0.85: 0.0005,
            # 0.88: 0.0005,
            # 0.89: 0.0005,
            # 0.9:  0.0005,
            # 0.91: 0.0005,
            # 0.92: 0.0005
            0.6:  0.005,
            0.7:  0.001,
            0.8:  0.0005,
            0.85: 0.0001,
            0.88: 0.00005,
            0.89: 0.00001,
            0.9:  0.000005,
            0.91: 0.000001,
            0.92: 0.0000005
        }
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
        - n_epochs (int): Total number of training epochs (default: 5000).
        - report_frequency (int): Frequency of reporting model metrics (AUC and Accuracy) during training (default: 1).
        - auc_threshold (float): AUC threshold for early stopping (default: 0.9).
        - clipnorm (float): Gradient clipping norm to prevent exploding gradients (default: 2.0).
        - simplify_categories (bool): Whether to simplify categories in the dataset (default: True).
        - holdout_size (float): Proportion of samples withheld during training (default: 0.5).
        - multiplier (int): Scales the number of nodes in most network layers (default: 3).
        - auc_thresold (list): auc values for the test set for which the learning rate should be adjusted
        - lr_dict (dict): Dictionary defining the learning rate based on the measured test set accuracy.

        Raises:
        - ValueError: If `current_genes` is not a non-empty list.
        """
        if not isinstance(current_genes, list) or not current_genes:
            raise ValueError(
                "The 'current_genes' parameter must be a non-empty list of genes.")

        self.input_data = input_data  # RcDataPreparation class object for training the model
        # List of genes provided by the user to define model features.
        self.current_genes = current_genes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate  # Dropout rate for regularization.
        # Balance technology and outcomes during training.
        self.balance = balance
        self.l2_reg = l2_reg  # Degree of L2 regularization.
        self.batch_size = batch_size  # Batch size for training.
        self.n_epochs = n_epochs  # Total number of training epochs.
        # Frequency for collecting metrics during training.
        self.report_frequency = report_frequency
        self.auc_threshold = auc_threshold  # AUC threshold for early stopping.
        # Gradient clipping value to prevent exploding gradients.
        self.clipnorm = clipnorm
        # Whether to reduce data categories (e.g., microarray vs. sequencing).
        self.simplify_categories = simplify_categories
        # Proportion of samples withheld during training.
        self.holdout_size = holdout_size
        # Scales the number of nodes in most layers of the network.
        self.multiplier = multiplier
        # AUC values at which the learning rate should be adjusted
        self.auc_thresholds = auc_thresholds
        # Dynamically adjusts the learning rate based on the test set accuracy
        self.lr_dict = lr_dict
        self.outcome_classifier = None  # ANN model which is instantiated and trained
        self.test_accuracy_list = []  # list of metrics for evaluating the model
        self.train_accuracy_list = []  # list of metrics for evaluating the model
        self.test_auc_list = []  # list of metrics for evaluating the model
        self.train_auc_list = []  # list of metrics for evaluating the model

        # automatically executed functions for establishin and training the model
        self.set_mixed_precision()
        self.subset_input_data()
        self.build_outcome_classifier()
        self.train_the_model()

    def set_mixed_precision(self):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    def subset_input_data(self):
        """
        Subsets the data during training.
        """
        # prior retrival of gene set indices was too slow
        # gene_set_indices = [i for i, item in enumerate(self.input_data.genes_list) if item in self.current_genes]
        current_genes_set = set(self.current_genes)  # O(M)
        gene_set_indices = [i for i, item in enumerate(
            self.input_data.genes_list) if item in current_genes_set]  # O(N)
        self.x_train = self.input_data.x_train[:, gene_set_indices]
        self.x_test = self.input_data.x_test[:, gene_set_indices]

    def build_outcome_classifier(self):
        """
        Establishes the model.
        """
        self.outcome_classifier = tf.keras.Sequential()
        # Input shape matches your data
        self.outcome_classifier.add(
            layers.Input(shape=(len(self.current_genes),)))

        self.outcome_classifier.add(layers.Dense(
            (512*self.multiplier), kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        # Leaky ReLU helps with vanishing gradients
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        # self.outcome_classifier.add(layers.Dropout(self.dropout_rate))

        self.outcome_classifier.add(layers.Dense(
            (256*self.multiplier), kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        # self.outcome_classifier.add(layers.Dropout(self.dropout_rate))

        self.outcome_classifier.add(layers.Dense(
            (128*self.multiplier), kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        self.outcome_classifier.add(layers.Dropout(self.dropout_rate))

        self.outcome_classifier.add(layers.Dense(
            (64*self.multiplier), kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        self.outcome_classifier.add(layers.Dropout(self.dropout_rate))

        self.outcome_classifier.add(layers.Dense(
            (32), kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))

        # Output layer for binary classification with sigmoid activation
        self.outcome_classifier.add(layers.Dense(1, activation='sigmoid'))

    def train_the_model(self):
        """
        Trains the model.
        """

        # # set up the optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=self.clipnorm)
        # determine the sample and batch size
        # number of samples used in each training epoch
        n_samples = math.floor(self.x_train.shape[0] * (1-self.holdout_size))
        # Calculate the number of steps per epoch
        n_steps_per_epoch = n_samples // self.batch_size
        # Compile the outcome discriminator
        self.outcome_classifier.compile(
            optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Training loop
        for epoch in range(self.n_epochs):
            total_loss = 0.0  # To accumulate losses
            total_accuracy = 0.0  # To accumulate accuracy
            # Initialize gradient accumulator
            accumulated_grads = [tf.zeros_like(
                var) for var in self.outcome_classifier.trainable_variables]

            # Split train data randomly, holding out a portion for generalization
            x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(
                self.x_train, self.input_data.y_train, test_size=self.holdout_size, random_state=None)

            # Mini-batch training loop
            for step in range(n_steps_per_epoch):
                # Balance batches if necessary
                batch_indices = []

                if self.balance:
                    # Get unique class labels
                    unique_classes = np.unique(y_train_temp)

                    # Ensure each class is represented equally in the batch
                    for class_label in unique_classes:
                        condition_indices = np.where(y_train_temp == class_label)[
                            0]  # Get indices for this class
                        condition_batch_indices = np.random.choice(condition_indices,
                                                                   size=self.batch_size // len(
                                                                       unique_classes),
                                                                   replace=True)  # Sample with replacement if needed
                        batch_indices.append(condition_batch_indices)

                    # Merge class-specific batches
                    batch_indices = np.concatenate(batch_indices)
                else:
                    batch_indices = np.random.choice(
                        len(x_train_temp), size=self.batch_size, replace=True)

                # X_batch = X_train_temp[np.concatenate(batch_indices)]

                x_batch = x_train_temp[batch_indices if isinstance(
                    batch_indices, list) else np.array(batch_indices)]

                # y_batch = y_train_temp[np.concatenate(batch_indices)]
                # y_batch = y_train_temp.iloc[np.concatenate(batch_indices)]

                # y_batch = y_train_temp[np.concatenate(batch_indices)]
                y_batch = y_train_temp[np.concatenate(batch_indices)] if isinstance(
                    batch_indices, list) else y_train_temp[batch_indices]

                # Adjust labels shape for binary_crossentropy
                y_batch = tf.expand_dims(y_batch, axis=-1)

                # Perform the training step and collect gradients
                outcome_loss, accuracy, classifier_grads = train_step(
                    self.outcome_classifier, x_batch, y_batch)

                # Accumulate gradients and losses
                total_loss += outcome_loss.numpy()
                total_accuracy += accuracy.numpy()
                accumulated_grads = [acc_grad + grad for acc_grad,
                                     grad in zip(accumulated_grads, classifier_grads)]

            # Average the accumulated gradients
            averaged_grads = [
                grad / n_steps_per_epoch for grad in accumulated_grads]

            # Apply averaged gradients to update model weights
            optimizer.apply_gradients(
                zip(averaged_grads, self.outcome_classifier.trainable_variables))

            # Calculate average loss and accuracy for the epoch
            avg_loss = total_loss / n_steps_per_epoch
            avg_accuracy = total_accuracy / n_steps_per_epoch

            if epoch % self.report_frequency == 0:

                # Evaluate on training data
                train_predictions = self.outcome_classifier(self.x_train)
                # Reshape to match logits shape
                train_labels = tf.expand_dims(self.input_data.y_train, axis=-1)
                train_labels_float = tf.cast(train_labels, tf.float32)

                # Compute Training AUC
                train_predictions_np = train_predictions.numpy().flatten()
                train_labels_np = train_labels_float.numpy().flatten()
                train_auc = roc_auc_score(
                    train_labels_np, train_predictions_np)

                # Compute Training Accuracy
                predicted_train_labels = tf.cast(
                    train_predictions > 0.5, tf.float32)  # Threshold at 0.5
                train_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                    predicted_train_labels, train_labels_float), tf.float32)
                ).numpy()

                # Evaluate on test data
                outcome_predictions = self.outcome_classifier(self.x_test)
                # Reshape to match logits shape
                outcome_labels = tf.expand_dims(
                    self.input_data.y_test, axis=-1)
                outcome_labels_float = tf.cast(outcome_labels, tf.float32)

                # Compute Test AUC
                outcome_predictions_np = outcome_predictions.numpy().flatten()
                outcome_labels_np = outcome_labels_float.numpy().flatten()
                test_auc = roc_auc_score(
                    outcome_labels_np, outcome_predictions_np)

                # Compute Test Accuracy
                predicted_outcome_labels = tf.cast(
                    outcome_predictions > 0.5, tf.float32)
                test_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                    predicted_outcome_labels, outcome_labels_float), tf.float32)
                ).numpy()

                # Store metrics
                self.train_auc_list.append(train_auc)  # Store train AUC
                self.test_auc_list.append(test_auc)
                self.train_accuracy_list.append(train_accuracy)
                self.test_accuracy_list.append(test_accuracy)

                # Print status every 100 * report_frequency epochs
                if epoch % (self.report_frequency * 10) == 0:
                    print(f'Epoch {epoch}, '
                          f'Avg Outcome Loss: {avg_loss:.4f}, '
                          f'Train AUC: {train_auc:.4f}, '
                          f'Train Accuracy: {train_accuracy:.4f}, '
                          f'Test AUC: {test_auc:.4f}, '
                          f'Test Accuracy: {test_accuracy:.4f}')

                # Adjust learning rate based on test AUC
                adjust_learning_rate_by_auc(epoch, self.outcome_classifier, x_test_temp,
                                            y_test_temp,
                                            self.lr_dict, self.auc_thresholds, test_auc)

                # Early stopping condition
                if test_auc > self.auc_threshold or epoch > self.n_epochs:
                    print('Early stopping triggered based on AUC')
                    break
