# src/utils/plotting_utils.py
import logging
import os
from typing import Dict, List

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DeadNeuronPlotter:
    """Handles plotting of dead neuron statistics."""

    def __init__(self, top_n_layers: int = 10, threshold: float = 1e-5, output_dir: str = None, track_interval: int = 100):
        """
        Initializes the plotter.

        Args:
            top_n_layers: The number of layers with the highest max percentage
                          of dead neurons to include in the plot.
        """
        self.top_n_layers = top_n_layers
        self.threshold = threshold
        self.output_dir = output_dir
        self.track_interval = track_interval
        # Suppress matplotlib font warnings if desired - moved to train.py for main process check
        # mpl_logger = logging.getLogger('matplotlib.font_manager')
        # mpl_logger.setLevel(logging.WARNING)

    def plot_heatmap(self, weights_history, save_dir):
        for layer in weights_history:
            last_epoch_weights = weights_history[layer][-1]
            weights = np.array(last_epoch_weights)

            collapsed = np.mean(np.abs(weights), axis=1)

            num_filters = collapsed.shape[0]
            filter_h, filter_w = collapsed.shape[1], collapsed.shape[2]
            grid_h = int(np.ceil(np.sqrt(num_filters)))
            grid_w = int(np.ceil(num_filters / grid_h))

            heatmap = np.zeros((grid_h * filter_h, grid_w * filter_w))

            for idx in range(num_filters):
                row = idx // grid_w
                col = idx % grid_w
                heatmap[
                row * filter_h:(row + 1) * filter_h,
                col * filter_w:(col + 1) * filter_w
                ] = collapsed[idx]

            plt.figure(figsize=(10, 10))
            plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
            plt.title(f"Filter Weight Heatmap - Last Epoch - {layer}")
            plt.axis('off')
            plt.colorbar(label='Mean |Weight|')

            # plt.show()
            self.save_and_close(os.path.join(save_dir, f'heatmap_{layer}.png'))

    def plot_dead_over_steps(self, weights_history, save_dir):
        for layer in weights_history:
            weights_history_layer = weights_history[layer]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for idx, weights in enumerate(weights_history_layer):
                weights = np.array(weights)

                dead_mask = np.min(np.abs(weights), axis=1) < self.threshold
                f_idx, i_idx, j_idx = np.where(dead_mask)
                z_idx = np.full_like(f_idx, idx * self.track_interval)
                ax.scatter(i_idx, j_idx, z_idx, color='red', s=10)

            ax.set_xlabel('Filter Height (x)')
            ax.set_ylabel('Filter Width (y)')
            ax.set_zlabel('Epoch (z)')
            ax.set_title('Dead Weights in Filters Over Epochs')

            # plt.show()
            self.save_and_close(os.path.join(save_dir, f'dead_weights_3d_{layer}.png'))

    def plot_history(self, percent_history: Dict[str, List[float]], save_path: str, csv_path: str, xlabel: str = "Step"):
        """
        Plots the percentage of dead neurons over training steps for tracked layers and saves the plot.

        Args:
            percent_history: Dict mapping layer names to lists of percentages per track interval.
            save_path: Path to save the generated plot image.
            csv_path: Path to save the data frame.
        """
        if not percent_history:
            logger.warning("No dead neuron history collected, skipping plot generation.")
            return

        records = []
        max_len = 0
        # Convert each epoch to corresponding step (epoch * track_interval)
        for layer, history in percent_history.items():
            max_len = max(max_len, len(history))
            for i, percentage in enumerate(history):
                records.append({
                    "step": i * self.track_interval,
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
                .max()
                .sort_values(ascending=False)
                .head(self.top_n_layers)
                .index
            )
            logger.info(f"Plotting top {self.top_n_layers} layers with highest max dead neuron percentage.")
        else:
            top_layers = df['layer'].unique()
            logger.info(f"Plotting all {len(top_layers)} tracked layers.")

        # Save the dataframe to CSV
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(15, 7))
        step_range = [i * self.track_interval for i in range(max_len)]

        for layer in top_layers:
            layer_data = df[df["layer"] == layer].set_index("step")
            layer_data = layer_data.reindex(step_range)
            plt.plot(layer_data.index, layer_data["percentage"], label=layer, marker='.', linestyle='-')

        plt.xlabel(xlabel)
        tick_frequency = max(self.track_interval, (step_range[-1] // 15 if len(step_range) > 15 else self.track_interval))
        plt.xticks(range(0, step_range[-1] + 1, tick_frequency))
        plt.ylabel(f"% of weights < {self.threshold:.1e}")
        plt.title("Percentage of Near-Zero Weights Over Time (Steps)")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        self.save_and_close(save_path)

    def plot_matrix(self, percent_history, save_path, xlabel="Step"):
        if not percent_history:
            return
        layers = list(percent_history.keys())
        steps = sorted({x for hist in percent_history.values() for x, _ in hist})
        matrix = np.full((len(layers), len(steps)), np.nan)
        for li, layer in enumerate(layers):
            for x, y in percent_history[layer]:
                si = steps.index(x)
                matrix[li, si] = y
        plt.figure(figsize=(12, max(3, 0.3 * len(layers))))
        plt.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label=f"% |w|<{self.threshold:.1e}")
        plt.yticks(range(len(layers)), layers, fontsize=8)
        plt.xticks(np.linspace(0, len(steps) - 1, 10, dtype=int),
                   [steps[i] for i in np.linspace(0, len(steps) - 1, 10, dtype=int)])
        plt.xlabel(xlabel)
        plt.ylabel("Layer")
        plt.title("Dead-weight heat-map")
        self.save_and_close(save_path)

    def save_and_close(self, save_path):
        try:
            plt.savefig(save_path, bbox_inches='tight')  # Use bbox_inches='tight' with tight_layout
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}")
        finally:
            plt.close()  # Ensure figure is closed even if saving fails

    def plot_all(self, percent_history, weights_history) -> None:
        plot_path = os.path.join(self.output_dir, "dead_neuron_percentage_history.png")
        csv_path = os.path.join(self.output_dir, "dead_neuron_percentage_history.csv")
        self.plot_history(percent_history, plot_path, csv_path)
        self.plot_dead_over_steps(weights_history, self.output_dir)
        self.plot_heatmap(weights_history, self.output_dir)
        logger.info(f"Saved dead neuron history plot to {plot_path}")


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
    plotter = DeadNeuronPlotter(top_n_layers=5, threshold=1e-5)
    plotter.plot_history(dummy_history, save_path="./dummy_dead_neuron_plot.png")
    print("Dummy plot generated (check dummy_dead_neuron_plot.png)")

