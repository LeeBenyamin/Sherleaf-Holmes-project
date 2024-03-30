import pandas as pd
import matplotlib.pyplot as plt
import os
import random

def plot_results(file_path, save_path):
    """
    Plot results from the DataFrame.

    Parameters:
    - file_path of csv to plot
    - save_path: Optional, path to save the plot as an image file
    """
    plt.figure(figsize=(10, 8))
    df = pd.read_csv(file_path)

    # Generate random colors
    colors = [(random.random(), random.random(), random.random()) for _ in range(len(df))]

    # Scatter plot with markers of varying sizes and colors
    for i in range(len(df)):
        plt.scatter(df['Training time (minutes)'][i], df['test accuracy'][i], label=df['net'][i], s=50, color=colors[i])

    plt.grid()
    plt.title('Transfer learning for Plant Disease', fontsize=26)

    # Customize labels and legend
    plt.xlabel('Training time (min)', fontsize=20)
    plt.ylabel('Test accuracy', fontsize=20)
    plt.legend(loc='upper right', fontsize=14)

    plot_filename = 'all_net' + '_plot.png'
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()  # Close the plot to release memory


def plot_from_csv(csv_dir, save_dir, value):
    """
    Generate accuracy plots from CSV files.

    Parameters:
        csv_dir (str): Directory containing CSV files.
        save_dir (str): Directory to save generated plots.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            # Read the CSV file
            filepath = os.path.join(csv_dir, filename)
            data = pd.read_csv(filepath)
            plot_title = filename.replace('results_', '').replace('_', ' ')
            plot_title = plot_title.replace('.csv', '')

            # Extract data
            epochs = data['epoch'].unique()  # Get unique epochs
            train_accuracy = data[data['phase'] == 'train'][value]
            val_accuracy = data[data['phase'] == 'val'][value]

            train_accuracy = train_accuracy[:len(epochs)]
            val_accuracy = val_accuracy[:len(epochs)]

            # Plotting
            plt.figure(figsize=(10, 5))

            # Plot for training accuracy
            plt.plot(epochs, train_accuracy, marker='o', linestyle='-', color='b', label='Train ' + value)

            # Plot for validation accuracy
            plt.plot(epochs, val_accuracy, marker='o', linestyle='-', color='r', label='Validation ' + value)

            plt.title(plot_title)
            plt.xlabel('Epochs')
            plt.ylabel(value)
            plt.grid(True)
            plt.legend()

            # Save the plot
            plot_filename = plot_title + '_' + value + '_plot.png'
            plt.savefig(os.path.join(save_dir, plot_filename))
            plt.close()  # Close the plot to release memory

# Example usage:
csv_directory = r"transfer learning results/csv"
save_directory = r"transfer learning results/graphs"
plot_from_csv(csv_directory, save_directory, 'loss')

# Example usage:
file_path = r"transfer learning results/csv"
save_path = r"transfer learning results/graphs"
plot_results(file_path, save_path)
