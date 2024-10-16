import matplotlib.pyplot as plt
import numpy as np
import re

def visualize_training_logs(filename):
    epochs = []
    losses = []
    val_losses = []

    # Read log data from file
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Match and extract epoch, loss, and validation loss using regex
                match = re.match(r'Epoch (\d+): Loss:\s*([\d.]+), Validation Loss:\s*([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    val_loss = float(match.group(3))

                    epochs.append(epoch)
                    losses.append(loss)
                    val_losses.append(val_loss)
                else:
                    print(f"Line format not recognized: {line.strip()}")
            except ValueError as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")
                continue

    # Use logarithmic normalization to handle different scales
    def log_normalize(values):
        if not values:
            print("Warning: No values provided for normalization.")
            return []

        min_val = min(values)
        values_shifted = np.array(values) + abs(min_val) + 1e-10  # Shift to positive and avoid log(0)
        log_values = np.log(values_shifted)
        return (log_values - np.min(log_values)) / (np.max(log_values) - np.min(log_values))

    # Normalize losses
    normalized_losses = log_normalize(losses)
    normalized_val_losses = log_normalize(val_losses)

    # Plot training and validation loss
    plt.figure(figsize=(14, 7))
    plt.plot(epochs, normalized_losses, label='Normalized Training Loss', color='blue', linestyle='-', marker='o', markersize=4)
    plt.plot(epochs, normalized_val_losses, label='Normalized Validation Loss', color='orange', linestyle='--', marker='x', markersize=4)

    # Highlight minimum validation loss point
    if val_losses:  # Ensure there are validation losses to work with
        min_val_loss_idx = val_losses.index(min(val_losses))
        min_val_loss_normalized = normalized_val_losses[min_val_loss_idx]
        plt.annotate(f'Min Val Loss: {min_val_loss_normalized:.2f}',
                     xy=(epochs[min_val_loss_idx], min_val_loss_normalized),
                     xytext=(epochs[min_val_loss_idx] + 1, min_val_loss_normalized + 0.1),
                     arrowprops=dict(facecolor='blue', arrowstyle='->', lw=1.5),
                     fontsize=10, color='blue')

    # Highlight maximum validation loss point
    if val_losses:  # Ensure there are validation losses to work with
        max_val_loss_idx = val_losses.index(max(val_losses))
        max_val_loss_normalized = normalized_val_losses[max_val_loss_idx]
        plt.annotate(f'Max Val Loss: {max_val_loss_normalized:.2f}',
                     xy=(epochs[max_val_loss_idx], max_val_loss_normalized),
                     xytext=(epochs[max_val_loss_idx] + 1, max_val_loss_normalized + 0.1),
                     arrowprops=dict(facecolor='orange', arrowstyle='->', lw=1.5),
                     fontsize=10, color='orange')

    # Set y-limits for better visualization
    plt.ylim(0, 1.1)

    loss_function_run = filename.strip("log_").rsplit('.',1)[0]
    
    loss_functions = {"schrodinger" : "Schrödinger",
                      "navier_stokes" : "Navier-Stokes",
                      "heat" : "Heat",
                      "wave" : "Wave",
                      "maxwell" : "Maxwell"}

    loss_function = re.sub('_[0-9]+$', '',loss_function_run)

    # Labeling the plot
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Log-Normalized Loss', fontsize=14)
    plt.title(loss_functions[loss_function] + ' Log-Normalized Training and Validation Loss over Epochs ' + loss_function_run.strip(loss_function + "_"), fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('log_normalized_training_validation_loss_' + loss_function_run + '.png', dpi=300)

    # Show the plot
    plt.show()

# Example usage:
visualize_training_logs('log_schrodinger.txt')
