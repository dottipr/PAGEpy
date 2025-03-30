import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras import mixed_precision


# should increase model efficiency
# not cpu compatible, commented out for now

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

def adjust_learning_rate_by_auc(epoch, model, x_test, y_test_outcome, lr_dict, auc_thresholds, test_auc):
    # Get model's current learning rate
    current_lr = tf.keras.backend.get_value(model.optimizer.lr)

    # Adjust learning rate based on AUC thresholds
    new_lr = current_lr
    for threshold in auc_thresholds:
        if test_auc >= threshold:
            new_lr = lr_dict[threshold]
    
    # Set new learning rate
    if new_lr != current_lr:
        tf.keras.backend.set_value(model.optimizer.lr, new_lr)
        #print(f"Epoch {epoch + 1}: Adjusting learning rate to {new_lr:.6f}")
    
    return test_auc

class PredAnnModel:
    def __init__(
        self,
        input_data,
        current_genes,
        learning_rate = 0.01,
        dropout_rate=0.3,
        balance=True,
        l2_reg=0.2,
        batch_size=64,
        num_epochs=5000,
        report_frequency=1,
        auc_threshold=0.95,
        clipnorm=2.0,
        simplify_categories=True,
        holdout_size=0.5,
        multiplier=3,
        auc_thresholds = [0.6, 0.7, 0.8, 0.85, 0.88,0.89,0.90,0.91,0.92],
        lr_dict = {
                    0.6:  0.005,
                    0.7:  0.001,
                    0.8:  0.0005,
                    0.85: 0.0005,
                    0.88: 0.0005,
                    0.89: 0.0005,
                    0.9:  0.0005,
                    0.91: 0.0005,
                    0.92: 0.0005
                    # 0.6:  0.005,
                    # 0.7:  0.001,
                    # 0.8:  0.0005,
                    # 0.85: 0.0001,
                    # 0.88: 0.00005,
                    # 0.89: 0.00001,
                    # 0.9:  0.000005,
                    # 0.91: 0.000001,
                    # 0.92: 0.0000005
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
        - num_epochs (int): Total number of training epochs (default: 5000).
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
            raise ValueError("The 'current_genes' parameter must be a non-empty list of genes.")

        self.input_data = input_data  # RcDataPreparation class object for training the model
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
        self.holdout_size = holdout_size  # Proportion of samples withheld during training.
        self.multiplier = multiplier  # Scales the number of nodes in most layers of the network.
        self.auc_thresholds = auc_thresholds  # AUC values at which the learning rate should be adjusted
        self.lr_dict =  lr_dict  # Dynamically adjusts the learning rate based on the test set accuracy
        self.outcome_classifier = None  # ANN model which is instantiated and trained
        self.test_accuracy_list = []  # list of metrics for evaluating the model
        self.train_accuracy_list = []  # list of metrics for evaluating the model
        self.test_auc_list = []  # list of metrics for evaluating the model
        self.train_auc_list = []  #list of metrics for evaluating the model
        #self.train_auc_list = []  # list of metrics for evaluating the model
        self.current_epoch_list = []  # list of epoch numbers for trackking the metrics across models

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
        gene_set_indices = [i for i, item in enumerate(self.input_data.genes_list) if item in current_genes_set]  # O(N)
        self.x_train = self.input_data.x_train[:, gene_set_indices]
        self.x_test = self.input_data.x_test[:, gene_set_indices]

    def build_outcome_classifier(self):
        """
        Establishes the model.
        """
        self.outcome_classifier = keras.Sequential()
        self.outcome_classifier.add(layers.Input(shape=(len(self.current_genes),)))  # Input shape matches your data
        
        self.outcome_classifier.add(layers.Dense((512*self.multiplier),kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))  # Leaky ReLU helps with vanishing gradients
        # self.outcome_classifier.add(layers.Dropout(self.dropout_rate))
    
        self.outcome_classifier.add(layers.Dense((256*self.multiplier),kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        # self.outcome_classifier.add(layers.Dropout(self.dropout_rate))
    
        self.outcome_classifier.add(layers.Dense((128*self.multiplier),kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        self.outcome_classifier.add(layers.Dropout(self.dropout_rate))
        
        self.outcome_classifier.add(layers.Dense((64*self.multiplier),kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        self.outcome_classifier.add(layers.Dropout(self.dropout_rate))
    
        self.outcome_classifier.add(layers.Dense((32),kernel_initializer='he_normal'))
        self.outcome_classifier.add(layers.BatchNormalization())
        self.outcome_classifier.add(layers.LeakyReLU(alpha=0.1))
        
        # Output layer for binary classification with sigmoid activation
        self.outcome_classifier.add(layers.Dense(1, activation='sigmoid'))

    def train_the_model(self):
        """
        Trains the model.
        """
        
        # # set up the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm = self.clipnorm)
        # determine the sample and batch size
        num_samples = math.floor(self.x_train.shape[0]* (1-self.holdout_size))  # number of samples used in each training epoch        
        # Calculate the number of steps per epoch
        num_steps_per_epoch = num_samples // self.batch_size
        # Compile the outcome discriminator
        self.outcome_classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0.0  # To accumulate losses
            total_accuracy = 0.0  # To accumulate accuracy
            accumulated_grads = [tf.zeros_like(var) for var in self.outcome_classifier.trainable_variables]  # Initialize gradient accumulator
        
            # Split train data randomly, holding out a portion for generalization
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(self.x_train, self.input_data.y_train, test_size=self.holdout_size, random_state=None)
            
            # Mini-batch training loop
            for step in range(num_steps_per_epoch):
                # Balance batches if necessary
                batch_indices = []
                
                if self.balance:
                    # Get unique class labels
                    unique_classes = np.unique(y_train_temp)

                    # Ensure each class is represented equally in the batch
                    for class_label in unique_classes:
                        condition_indices = np.where(y_train_temp == class_label)[0]  # Get indices for this class
                        condition_batch_indices = np.random.choice(condition_indices, 
                                                                   size=self.batch_size // len(unique_classes), 
                                                                   replace=True)  # Sample with replacement if needed
                        batch_indices.append(condition_batch_indices)

                    batch_indices = np.concatenate(batch_indices)  # Merge class-specific batches
                else:
                    batch_indices = np.random.choice(len(X_train_temp), size=self.batch_size, replace=True)
                
                # X_batch = X_train_temp[np.concatenate(batch_indices)]
                
                X_batch = X_train_temp[batch_indices if isinstance(batch_indices, list) else np.array(batch_indices)]

                
                #y_batch = y_train_temp[np.concatenate(batch_indices)]
                # y_batch = y_train_temp.iloc[np.concatenate(batch_indices)]
                
                # y_batch = y_train_temp[np.concatenate(batch_indices)]
                y_batch = y_train_temp[np.concatenate(batch_indices)] if isinstance(batch_indices, list) else y_train_temp[batch_indices]


                
                y_batch = tf.expand_dims(y_batch, axis=-1)  # Adjust labels shape for binary_crossentropy
                        
                # Perform the training step and collect gradients
                outcome_loss, accuracy, classifier_grads = train_step(self.outcome_classifier,X_batch, y_batch)
                
                # Accumulate gradients and losses
                total_loss += outcome_loss.numpy()
                total_accuracy += accuracy.numpy()
                accumulated_grads = [acc_grad + grad for acc_grad, grad in zip(accumulated_grads, classifier_grads)]
        
            # Average the accumulated gradients
            averaged_grads = [grad / num_steps_per_epoch for grad in accumulated_grads]
        
            # Apply averaged gradients to update model weights
            optimizer.apply_gradients(zip(averaged_grads, self.outcome_classifier.trainable_variables))
        
            # Calculate average loss and accuracy for the epoch
            avg_loss = total_loss / num_steps_per_epoch
            avg_accuracy = total_accuracy / num_steps_per_epoch
        
            if epoch % self.report_frequency == 0:
    
                # Evaluate on training data
                train_predictions = self.outcome_classifier(self.x_train)
                train_labels = tf.expand_dims(self.input_data.y_train, axis=-1)  # Reshape to match logits shape
                train_labels_float = tf.cast(train_labels, tf.float32)

                # Compute Training AUC
                train_predictions_np = train_predictions.numpy().flatten()
                train_labels_np = train_labels_float.numpy().flatten()
                train_auc = roc_auc_score(train_labels_np, train_predictions_np)

                # Compute Training Accuracy
                predicted_train_labels = tf.cast(train_predictions > 0.5, tf.float32)  # Threshold at 0.5
                train_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_train_labels, train_labels_float), tf.float32))

                # Evaluate on test data
                outcome_predictions = self.outcome_classifier(self.x_test)
                outcome_labels = tf.expand_dims(self.input_data.y_test, axis=-1)  # Reshape to match logits shape
                outcome_labels_float = tf.cast(outcome_labels, tf.float32)

                # Compute Test AUC
                outcome_predictions_np = outcome_predictions.numpy().flatten()
                outcome_labels_np = outcome_labels_float.numpy().flatten()
                test_auc = roc_auc_score(outcome_labels_np, outcome_predictions_np)

                # Compute Test Accuracy
                predicted_outcome_labels = tf.cast(outcome_predictions > 0.5, tf.float32)
                test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_outcome_labels, outcome_labels_float), tf.float32))

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
                adjust_learning_rate_by_auc(epoch, self.outcome_classifier, X_test_temp, 
                                            y_test_temp, 
                                            self.lr_dict, self.auc_thresholds, test_auc)

                # Early stopping condition
                if test_auc > self.auc_threshold or epoch > self.num_epochs:
                    print('Early stopping triggered based on AUC')
                    break
