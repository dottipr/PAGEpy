import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def evaluate_model(input_model,input_data):
    
    train_accuracy = float(max(input_model.train_accuracy_list).numpy())
    train_accuracy = round(train_accuracy, 2)
    print(f'max train accuracy: {train_accuracy}')
    print(f'max train auc: {round(max(input_model.train_auc_list),2)}')
    
    test_accuracy = float(max(input_model.test_accuracy_list).numpy())
    test_accuracy = round(test_accuracy, 2)
    print(f'max test accuracy: {test_accuracy}')
    print(f'max test auc: {round(max(input_model.test_auc_list),2)}')

    report_frequency = 1

    # # After training, plot the metrics

    frequency_counts = pd.Series(input_data.y_test).value_counts()
    test_chance_level = frequency_counts[0]/len(input_data.y_test)

    frequency_counts = pd.Series(input_data.y_train).value_counts()
    train_chance_level = frequency_counts[0]/len(input_data.y_train)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    x_values = np.arange(1, len(input_model.train_accuracy_list) + 1) * report_frequency

    # Plot train accuracy
    axs[0].plot(x_values, input_model.train_accuracy_list, label='Training Accuracy', color='blue')
    axs[0].axhline(train_chance_level, color='black',linestyle ='--')
    axs[0].set_title('Training set accuracy over epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Training Accuracy')
    axs[0].grid()
    axs[0].legend()

    # Plot train auc
    axs[1].plot(x_values, input_model.train_auc_list, label='Train AUC', color='blue')
    axs[1].set_title('Training set AUC over epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Train AUC')
    axs[1].grid()
    axs[1].legend()

    # Plot test accuracay
    axs[2].plot(x_values, input_model.test_accuracy_list, label='Test Accuracy', color='orange')
    axs[2].axhline(test_chance_level, color='black',linestyle ='--')
    axs[2].set_title('Test set accuracy over epochs')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Test Accuracy')
    axs[2].grid()
    axs[2].legend()

    # Plot test auc
    axs[3].plot(x_values, input_model.test_auc_list, label='Test AUC', color='orange')
    axs[3].set_title('Test set AUC over epochs')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Test AUC')
    axs[3].grid()
    axs[3].legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_pso_row_averages(df):
    """
    Plots the average, maximum, minimum, and standard deviation of each row in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing numerical values.
    """
    row_averages = df.mean(axis=1)  # Compute the average across each row
    row_std = df.std(axis=1)  # Compute the standard deviation across each row
    row_max = df.max(axis=1)  # Compute the maximum value across each row
    row_min = df.min(axis=1)  # Compute the minimum value across each row
    
    plt.figure(figsize=(10, 5))
    
    # Plot row averages
    plt.plot(row_averages, marker='o', linestyle='-', color='b', label='Average AUC')
    
    # Plot the shaded region representing the standard deviation
    plt.fill_between(range(len(df)), row_averages - row_std, row_averages + row_std, color='b', alpha=0.2, label='Standard Deviation')
    
    # Plot row maximum values
    plt.plot(row_max, marker='s', linestyle='--', color='r', label='Max AUC')
    
    # Plot row minimum values
    plt.plot(row_min, marker='d', linestyle='-.', color='g', label='Min AUC')
    
    plt.xlabel("Generation")
    plt.ylabel("AUC Values")
    plt.title("Feature Set Performance Across Generations")
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def hamming_distance(vector1, vector2):
    # Ensure both vectors are of the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    
    # Use zip to iterate over both vectors and sum the differences
    return sum(el1 != el2 for el1, el2 in zip(vector1, vector2))

def plot_hamming_distance(input_dict):
    """ Plots the average hamming distance between all members of each generation
    Takes a dictionary corresponding to a list of lists
    """
    hamming_averages = []
    gen = []
    num_keys = len(input_dict)
    x = list(range(0, num_keys))
    
    for i in x:
        current_list = input_dict[i]
        hamming_generation_values = []
        for p in range(len(current_list) - 1):
            relevant_comparisons = list(range(p, len(current_list)))
            for pop_member in relevant_comparisons:
                distance = hamming_distance(input_dict[i][p], input_dict[i][pop_member])
                hamming_generation_values.append(distance)
        current_hamming_average = sum(hamming_generation_values) / len(hamming_generation_values)
        hamming_averages.append(current_hamming_average)
        gen.append(i)

    # Plot gen averages for hamming distance
    plt.plot(hamming_averages, marker='o', linestyle='-', color='b', label='Average AUC')
    
    # Plot the shaded region representing the standard deviation
    
    plt.xlabel("Generation")
    plt.ylabel("Average AUC Values")
    plt.title("Average feature set hamming distance across generations")
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_sorted_frequencies(loaded_dict, loaded_df):
    """
    Plots the sorted normalized frequencies of the value 1 in the first and last dataset in loaded_dict.
    
    Parameters:
        loaded_dict (dict): Dictionary containing lists of data.
        loaded_df (DataFrame): DataFrame to determine the last index.
    """
    # Extract first and last lists from loaded_dict
    lists_1 = loaded_dict[0]
    lists_2 = loaded_dict[loaded_df.shape[0] - 1]
    
    value_to_check = 1
    
    # Function to compute sorted normalized frequencies
    def get_sorted_frequencies(lists, value):
        num_lists = len(lists)
        frequencies = [Counter(col)[value] / num_lists for col in zip(*lists)]
        return sorted(frequencies)
    
    # Compute sorted frequencies for both datasets
    sorted_frequencies_1 = get_sorted_frequencies(lists_1, value_to_check)
    sorted_frequencies_2 = get_sorted_frequencies(lists_2, value_to_check)
    
    # Plot both line plots on the same axes
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_frequencies_1, marker='o', linestyle='-', color='blue', label="First Gen")
    plt.plot(sorted_frequencies_2, marker='s', linestyle='--', color='red', label="Latest Gen")
    
    # Labels and title
    plt.xlabel("Sorted Index")
    plt.ylabel(f"Normalized Frequency of {value_to_check}")
    plt.title("Comparison of Normalized Frequencies")
    plt.ylim(0, 1)  # Since it's normalized
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  # Optional: add grid for readability
    
    # Show plot
    plt.show()
