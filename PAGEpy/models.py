import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


@dataclass
class TrainingConfig:
    """
    Configuration class for neural network training parameters.
    Using default params set in original PAGEpy code

    Parameters: (TODO update this)
    - learning_rate (float): intial learning rate of the model
    - input_data (GeneExpressionDataset class object): data for training the model that has been appropriately formattedd.
    - dropout_rate (float): Dropout rate to prevent overfitting (default: 0.3).
    - balance (bool): Whether to balance technology and outcome variables during training (default: True).
    - l2_reg (float): Strength of L2 regularization (default: -0.2).
    - batch_size (int): Batch size for training (default: 16).
    - n_epochs (int): Total number of training epochs (default: 5000).
    - report_frequency (int): Frequency of reporting model metrics (AUC and Accuracy) during training (default: 1).
    - auc_threshold (float): AUC threshold for early stopping (default: 0.9).
    - clipnorm (float): Gradient clipping norm to prevent exploding gradients (default: 2.0).
    - holdout_size (float): Proportion of samples withheld during training (default: 0.5).
    - multiplier (int): Scales the number of nodes in most network layers (default: 3).
    - lr_schedule (dict): Dictionary defining the learning rate based on the measured test set accuracy.

    Remark: Using field(default_factory=lambda: ...) ensures that each instance
    gets its own fresh copy of the mutable object:
    """
    learning_rate: float = 0.01
    dropout_rate: float = 0.3
    l2_reg: float = 0.2
    balance_classes: bool = True
    report_frequency: int = 1
    auc_threshold: float = 1  # no early stopping; was: 0.95
    clipnorm: float = 2.0
    holdout_size: float = 0.5
    multiplier: int = 3
    lr_schedule: Dict[float, float] = field(default_factory=lambda: {
        0.6: 0.005, 0.7: 0.001, 0.8: 0.0005, 0.85: 0.0001,
        0.88: 0.00005, 0.89: 0.00001, 0.9: 0.000005,
        0.91: 0.000001, 0.92: 0.0000005
    })  # used in 'adjust_learning_rate_by_auc'


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy and TensorFlow.

    TODO: random seeds are being set a bit all over the place -> fix this
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


class SimpleNN:
    """
    Simple neural network for binary classification of genomic data.

    This class implements a feedforward neural network with configurable
    architecture and training parameters for binary classification tasks.
    """

    def __init__(
        self,
        n_input_features: int,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize the SimpleNN Model.

        Args:
            n_input_features: Number of input features (genes)
            config: Training configuration object
        """
        self.config = config or TrainingConfig()  # uses AdvancedNN's default values
        self.n_input_features = n_input_features

        # Model components
        self.model: Optional[tf.keras.Model] = None
        self.optimizer: Optional[tf.keras.optimizers.Optimizer] = None

        # Training metrics
        self.training_history = {
            'train_loss': [], 'train_accuracy': [], 'train_auc': [],
            'test_accuracy': [], 'test_auc': []
        }

        self._build_model()

    def _build_model(self) -> None:
        """
        Build the neural network architecture.
        """
        l2_reg = tf.keras.regularizers.l2(self.config.l2_reg)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_input_features,)),

            tf.keras.layers.Dense(
                256, activation='relu',
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal',  # Good for ReLU activations
            ),

            tf.keras.layers.Dense(
                128, activation='relu',
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal',
            ),

            tf.keras.layers.Dense(
                64, activation='relu',
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal',
            ),
            # no L2 reg on smaller layer
            tf.keras.layers.Dense(
                32, activation='relu',
                kernel_initializer='he_normal',
            ),
            # Output layer: single unit with sigmoid for binary classification
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Configure optimizer and compile model
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=None  # no clipnorm in simple model
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',  # Standard loss for binary classification
            # metrics=['accuracy']
        )
        # print(f"Model created with {self.n_input_features} input features")

    @tf.function  # Compile for faster execution
    def _train_step(
        self, x_batch: tf.Tensor, y_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Execute a single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = tf.keras.losses.binary_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # accuracy = tf.keras.metrics.binary_accuracy(y_batch, predictions)

        return loss, gradients

    def _create_balanced_batch(
        self, x_data: np.ndarray, y_data: np.ndarray, batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a balanced batch with equal representation from each class."""
        batch_indices = []
        unique_classes = np.unique(y_data)
        samples_per_class = batch_size // len(unique_classes)

        for class_label in unique_classes:
            # Get indices for this class
            class_indices = np.where(y_data == class_label)[0]
            selected_indices = np.random.choice(
                class_indices,
                size=samples_per_class,
                replace=True  # Allow replacement if class is small
            )
            batch_indices.extend(selected_indices)

        batch_indices = np.array(batch_indices)
        return x_data[batch_indices], y_data[batch_indices]

    def _create_random_batch(
        self, x_data: np.ndarray, y_data: np.ndarray, batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a random batch without class balancing."""
        indices = np.random.choice(len(x_data), size=batch_size, replace=True)
        return x_data[indices], y_data[indices]

    def _evaluate_model(
            self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate model performance on given data."""
        predictions = self.model(x_data, training=False)

        # Calculate AUC
        auc_score = roc_auc_score(y_data, predictions.numpy().flatten())

        # Calculate accuracy
        predicted_labels = (predictions > 0.5).numpy().astype(float)
        accuracy = np.mean(predicted_labels.flatten() == y_data)

        return auc_score, accuracy

    def train(
        self, x_train: np.ndarray, y_train: np.ndarray,
        n_epochs: int, batch_size: int, seed: int = 42,
        x_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the neural network model.
        """
        set_random_seed(seed=seed)  # Ensure reproducible results

        n_steps_per_epoch = len(x_train) // batch_size

        for epoch in range(n_epochs):
            # Training phase
            total_loss = 0.0
            accumulated_gradients = [
                tf.zeros_like(var) for var in self.model.trainable_variables
            ]

            for _ in range(n_steps_per_epoch):
                #  Create batch
                if self.config.balance_classes:
                    x_batch, y_batch = self._create_balanced_batch(
                        x_data=x_train, y_data=y_train, batch_size=batch_size,
                    )
                else:
                    x_batch, y_batch = self._create_random_batch(
                        x_data=x_train, y_data=y_train, batch_size=batch_size
                    )

                # Convert to tensors with explicit dtype
                x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

                # Ensure y_batch has the right shape for binary classification
                # Model outputs (batch_size, 1), so y_batch should also be (batch_size, 1)
                if len(y_batch.shape) == 1:
                    y_batch = tf.expand_dims(y_batch, -1)

                # Training step
                loss, gradients = self._train_step(x_batch, y_batch)

                # Accumulate metrics and gradients
                total_loss += loss.numpy()
                accumulated_gradients = [
                    acc_grad + grad
                    for acc_grad, grad in zip(accumulated_gradients, gradients)
                ]

            # Apply the accumulated gradients
            averaged_gradients = [
                grad / n_steps_per_epoch for grad in accumulated_gradients
            ]
            self.optimizer.apply_gradients(
                zip(averaged_gradients, self.model.trainable_variables)
            )

            # Record training loss
            train_loss = total_loss / n_steps_per_epoch
            self.training_history['train_loss'].append(train_loss)

            # Evaluation phase
            if (1+epoch) % self.config.report_frequency == 0:
                # Evaluate on training data
                train_auc, train_acc = self._evaluate_model(x_train, y_train)
                self.training_history['train_accuracy'].append(train_acc)
                self.training_history['train_auc'].append(train_auc)

                # Evaluate on test data if provided
                if x_test is not None and y_test is not None:
                    test_auc, test_acc = self._evaluate_model(x_test, y_test)
                    self.training_history['test_auc'].append(test_auc)
                    self.training_history['test_accuracy'].append(test_acc)

                    # Print progress
                    if (1+epoch) % (self.config.report_frequency * 10) == 0:
                        print(f'Epoch {1+epoch}/{n_epochs}:')
                        print(
                            f'\tTrain Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}')
                        print(
                            f'\tTest AUC: {test_auc:.4f}, Acc: {test_acc:.4f}')

                    # Early stopping
                    if test_auc >= self.config.auc_threshold:
                        print(
                            f'Early stopping at epoch {epoch}: AUC threshold reached')
                        break

        return self.training_history

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.model(x_data, training=False).numpy()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate model and return AUC score."""
        auc_score, _ = self._evaluate_model(x_test, y_test)
        return auc_score


class AdvancedNN(SimpleNN):
    """
    Advanced neural network model that extends SimpleNN with more sophisticated
    architecture, holdout validation, and adaptive learning rate scheduling.
    """

    def _build_model(self):
        """
        Build the neural network architecture with more layers and batch normalization.
        """
        l2_reg = tf.keras.regularizers.l2(self.config.l2_reg)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_input_features,)),
            tf.keras.layers.Dense(
                512*self.config.multiplier,
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal'
            ),
            tf.keras.layers.BatchNormalization(),
            # Leaky ReLU for vanishing gradients
            tf.keras.layers.LeakyReLU(alpha=0.1),
            # tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(
                256*self.config.multiplier,
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            # tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(
                128*self.config.multiplier,
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(
                64*self.config.multiplier,
                kernel_regularizer=l2_reg,
                kernel_initializer='he_normal'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(32, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        # Configure optimizer and compile model
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.clipnorm
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            # metrics=['accuracy']
        )
        # print(f"Model created with {self.n_input_features} input features")

    def _adjust_learning_rate_by_auc(self, epoch: int, test_auc: float) -> None:
        """Adjust learning rate based on AUC thresholds."""
        current_lr = self.optimizer.learning_rate.numpy()
        new_lr = current_lr

        # Adjust learning rate based on AUC thresholds
        for auc_t, lr in self.config.lr_schedule.items():
            if test_auc >= auc_t:
                new_lr = lr
                break

        # Set new learning rate if changed
        if not np.isclose(new_lr, current_lr):
            self.model.optimizer.learning_rate = new_lr
            print(f"Epoch {epoch}: Adjusting learning rate to {new_lr:.6f}")

        return test_auc

    def train(  # TODO: vedere dove mettere valor di default di n_epochs e batch_size
        self, x_train: np.ndarray, y_train: np.ndarray,
        n_epochs: int = 500, batch_size: int = 64, seed: int = 42,
        x_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the neural network model.

        Args:
            x_train: Training features
            y_train: Training labels
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            x_test: Optional test features
            y_test: Optional test labels

        Returns:
            Dictionary containing training history
        """
        set_random_seed(seed=seed)  # Ensure reproducible results

        # Create holdout split for training
        n_samples_per_epoch = math.floor(
            len(x_train) * (1-self.config.holdout_size))
        n_steps_per_epoch = n_samples_per_epoch // batch_size

        for epoch in range(n_epochs):
            # Re-split the data each epoch for better generalization
            x_train_split, x_holdout, y_train_split, y_holdout = train_test_split(
                x_train, y_train,
                test_size=self.config.holdout_size,
                # random_state was: None; trying to make it random & reproducible
                random_state=seed+epoch if seed else None,
                stratify=y_train,
            )

            # Training phase
            total_loss = 0.0
            accumulated_gradients = [
                tf.zeros_like(var) for var in self.model.trainable_variables
            ]

            # Mini-batch training loop
            for _ in range(n_steps_per_epoch):
                # Create batch
                if self.config.balance_classes:
                    x_batch, y_batch = self._create_balanced_batch(
                        x_data=x_train_split, y_data=y_train_split, batch_size=batch_size
                    )

                else:
                    x_batch, y_batch = self._create_random_batch(
                        x_data=x_train_split, y_data=y_train_split, batch_size=batch_size
                    )

                # Convert to tensors
                x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)

                # Ensure proper shape for binary classification
                if len(y_batch.shape) == 1:
                    y_batch = tf.expand_dims(y_batch, -1)

                # Training step
                loss, gradients = self._train_step(x_batch, y_batch)

                # Accumulate gradients and losses
                total_loss += loss.numpy()
                # total_accuracy += accuracy.numpy()
                accumulated_gradients = [
                    acc_grad + grad
                    for acc_grad, grad in zip(accumulated_gradients, gradients)
                ]

            # Average the accumulated gradients
            averaged_gradients = [
                grad / n_steps_per_epoch for grad in accumulated_gradients
            ]
            self.optimizer.apply_gradients(
                zip(averaged_gradients, self.model.trainable_variables)
            )

            # Record training loss
            train_loss = total_loss / n_steps_per_epoch
            self.training_history['train_loss'].append(train_loss)

            # Evaluation phase
            if (1+epoch) % self.config.report_frequency == 0:
                # Evaluate on training data
                train_auc, train_acc = self._evaluate_model(x_train, y_train)
                self.training_history['train_accuracy'].append(train_acc)
                self.training_history['train_auc'].append(train_auc)

                # Evaluate on test data if provided
                if x_test is not None and y_test is not None:
                    test_auc, test_acc = self._evaluate_model(x_test, y_test)
                    self.training_history['test_auc'].append(test_auc)
                    self.training_history['test_accuracy'].append(test_acc)
                else:
                    print("Test data not provided, evaluating model on holdout data...")
                    test_auc, test_acc = self._evaluate_model(
                        x_holdout, y_holdout)
                    self.training_history['test_auc'].append(test_auc)
                    self.training_history['test_accuracy'].append(test_acc)

                # Print progress
                if (1+epoch) % (self.config.report_frequency * 10) == 0:
                    print(f'Epoch {1+epoch}/{n_epochs}:')
                    print(
                        f'\tTrain Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}')
                    print(f'\tTest AUC: {test_auc:.4f}, Acc: {test_acc:.4f}')

                # Adjust learning rate based on test AUC
                if hasattr(self.config, 'lr_schedule'):
                    self._adjust_learning_rate_by_auc(epoch, test_auc)

                # Early stopping
                if test_auc >= self.config.auc_threshold:
                    print(
                        f'Early stopping at epoch {epoch}: AUC threshold reached')
                    break

        return self.training_history
