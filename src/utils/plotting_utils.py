# src/utils/plotting_utils.py
import logging
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict # Keep defaultdict if used internally, though not strictly needed for plotting func sig

logger = logging.getLogger(__name__)

class DeadNeuronPlotter:
    """Handles plotting of dead neuron statistics."""

    def __init__(self, top_n_layers: int = 10):
        """
        Initializes the plotter.

        Args:
            top_n_layers: The number of layers with the highest max percentage
                          of dead neurons to include in the plot.
        """
        self.top_n_layers = top_n_layers
        # Suppress matplotlib font warnings if desired - moved to train.py for main process check
        # mpl_logger = logging.getLogger('matplotlib.font_manager')
        # mpl_logger.setLevel(logging.WARNING)


    def plot_history(self, percent_history: Dict[str, List[float]], threshold: float, save_path: str):
        """
        Plots the percentage of dead neurons over epochs for tracked layers and saves the plot.

        Args:
            percent_history: Dict mapping layer names to lists of percentages per epoch.
            threshold: The threshold used for defining 'dead'.
            save_path: Path to save the generated plot image.
        """
        if not percent_history:
            logger.warning("No dead neuron history collected, skipping plot generation.")
            return

        records = []
        num_epochs = 0
        # Determine the number of epochs and flatten the history data
        for layer, history in percent_history.items():
            num_epochs = max(num_epochs, len(history))
            for epoch, percentage in enumerate(history):
                records.append({
                    "epoch": epoch,
                    "layer": layer,
                    "percentage": percentage
                })

        if not records:
            logger.warning("No valid records found in percent_history, skipping plot.")
            return

        df = pd.DataFrame(records)

        # Select top N layers based on max percentage reached
        if len(df['layer'].unique()) > self.top_n_layers:
            top_layers = (
                df.groupby("layer")["percentage"]
                .max() # Use max percentage to select interesting layers
                .sort_values(ascending=False)
                .head(self.top_n_layers) # Plot top N layers
                .index
            )
            logger.info(f"Plotting top {self.top_n_layers} layers with highest max dead neuron percentage.")
        else:
            # If fewer unique layers than top_n_layers, plot all of them
            top_layers = df['layer'].unique()
            logger.info(f"Plotting all {len(top_layers)} tracked layers.")


        plt.figure(figsize=(15, 7)) # Wider figure for potentially many labels
        epochs = range(num_epochs) # Create epoch range for x-axis

        for layer in top_layers:
            # Get data for the layer, ensure it spans all epochs (pad with NaN if needed)
            layer_data = df[df["layer"] == layer].set_index("epoch")
            # Reindex to ensure we plot across the full epoch range, fill missing with NaN
            layer_data = layer_data.reindex(epochs)
            plt.plot(layer_data.index, layer_data["percentage"], label=layer, marker='.', linestyle='-') # Use dots and lines

        plt.xlabel("Epoch")
        # Adjust x-tick frequency based on number of epochs
        tick_frequency = max(1, num_epochs // 15 if num_epochs > 15 else 1) # Show ~15 ticks max
        plt.xticks(range(0, num_epochs, tick_frequency))
        plt.ylabel(f"% of weights < {threshold:.1e}") # Format threshold nicely
        plt.title("Percentage of Near-Zero Weights Over Time (Epochs)")
        # Place legend outside plot area to avoid overlap
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Adjust layout to make space for legend
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right boundary

        try:
            plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches='tight' with tight_layout
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}")
        finally:
             plt.close() # Ensure figure is closed even if saving fails


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    dummy_history = {
        'layer1.weight': [1.0, 0.8, 0.7, 0.75, 0.6],
        'layer2.weight': [0.1, 0.2, 0.15, 0.3, 0.25],
        'layer3.weight': [5.0, 4.0, 6.0, 5.5, 7.0],
        # Add more layers to test top_n selection
        'layer4.weight': [0.5, 0.6, 0.5, 0.7, 0.6],
        'layer5.weight': [0.2, 0.3, 0.2, 0.4, 0.3],
        'layer6.weight': [0.3, 0.4, 0.3, 0.5, 0.4],
        'layer7.weight': [0.4, 0.5, 0.4, 0.6, 0.5],
        'layer8.weight': [0.6, 0.7, 0.6, 0.8, 0.7],
        'layer9.weight': [0.7, 0.8, 0.7, 0.9, 0.8],
        'layer10.weight': [0.8, 0.9, 0.8, 1.0, 0.9],
        'layer11.weight': [0.9, 1.0, 0.9, 1.1, 1.0],
    }
    plotter = DeadNeuronPlotter(top_n_layers=5)
    plotter.plot_history(dummy_history, threshold=1e-5, save_path="./dummy_dead_neuron_plot.png")
    print("Dummy plot generated (check dummy_dead_neuron_plot.png)")

