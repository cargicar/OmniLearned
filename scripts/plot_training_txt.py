import matplotlib.pyplot as plt
import re

def plot_losses(log_file_path):
    """
    Parses a log file with multi-line loss output, extracts Gen Loss and Gen Val Loss,
    and plots them on a single graph.

    Args:
        log_file_path (str): The path to the log file.
    """
    epochs = []
    gen_losses = []
    gen_val_losses = []
    current_epoch = None

    try:
        with open(log_file_path, 'r') as file:
            # Regular expressions for matching the two different lines
            epoch_pattern = re.compile(r"Epoch \[(\d+)/\d+\]")
            loss_pattern = re.compile(r"Gen Loss: ([\d.]+), Gen Val Loss: ([\d.]+)")
            
            for line in file:
                # First, check for the epoch line
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                # If we have an epoch, check for the loss line on subsequent lines
                if current_epoch is not None:
                    loss_match = loss_pattern.search(line)
                    if loss_match:
                        gen_loss = float(loss_match.group(1))
                        gen_val_loss = float(loss_match.group(2))
                        
                        epochs.append(current_epoch)
                        gen_losses.append(gen_loss)
                        gen_val_losses.append(gen_val_loss)
                        
                        # Reset current_epoch to wait for the next epoch line
                        current_epoch = None

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return

    # Check if any data was successfully extracted
    if not epochs:
        print("No loss data found in the log file with the expected format.")
        return

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[2:], gen_losses[2:], label='Generation Loss', marker='o', linestyle='-')
    plt.plot(epochs[2:], gen_val_losses[2:], label='Generation Validation Loss', marker='x', linestyle='--')
    
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/training_vs_validation.png")
    plt.close()
if __name__ == "__main__":
    # Change this to the actual path of your log file
    log_file_path = "train_rf_sep_22.txt" 
    plot_losses(log_file_path)
