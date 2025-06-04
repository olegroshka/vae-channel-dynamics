# src/utils/plotting_utils.py
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

logger = logging.getLogger(__name__)


class DeadNeuronPlotter:
    """Handles plotting of dead neuron statistics from DeadNeuronTracker."""

    def __init__(self, top_n_layers: int = 10, threshold: float = 1e-5, output_dir: str = None):
        self.top_n_layers = top_n_layers
        self.threshold = threshold  # This is the threshold used by DeadNeuronTracker for its calculations
        self.output_dir = output_dir if output_dir else "."
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"DeadNeuronPlotter initialized. Output dir: {self.output_dir}")

    def plot_heatmap(self, weights_history: Dict[str, List[np.ndarray]], layer_name: str):
        if not weights_history or layer_name not in weights_history or not weights_history[layer_name]:
            logger.warning(f"No weight history for layer {layer_name} to plot heatmap.")
            return

        latest_weights_snapshot = weights_history[layer_name][0]  # weights_history stores a list with ONE snapshot

        if not isinstance(latest_weights_snapshot, np.ndarray) or latest_weights_snapshot.ndim < 2:
            logger.warning(
                f"Weights for layer {layer_name} are not suitable for heatmap. Shape: {latest_weights_snapshot.shape}")
            return

        fig = None
        if latest_weights_snapshot.ndim == 4:
            filter_magnitudes = np.mean(np.abs(latest_weights_snapshot), axis=(1, 2, 3))
            num_filters = filter_magnitudes.shape[0]
            if num_filters == 0: logger.warning(f"No filters found for heatmap in layer {layer_name}."); return

            fig, ax = plt.subplots(figsize=(10, max(5, num_filters * 0.2)))
            ax.bar(range(num_filters), filter_magnitudes, color='skyblue')
            ax.set_xlabel("Output Channel Index")
            ax.set_ylabel(f"Mean Abs Weight per Output Channel")
            ax.set_title(f"Filter Weight Magnitudes - Last Tracked Step - {layer_name}")
            plt.tight_layout()
            self.save_and_close(os.path.join(self.output_dir, f'filter_magnitudes_{layer_name.replace(".", "_")}.png'),
                                fig)

        elif latest_weights_snapshot.ndim == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(np.abs(latest_weights_snapshot), cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='Absolute Weight Value')
            ax.set_xlabel("Input Features")
            ax.set_ylabel("Output Features")
            ax.set_title(f"Weight Heatmap - Last Tracked Step - {layer_name}")
            plt.tight_layout()
            self.save_and_close(os.path.join(self.output_dir, f'heatmap_{layer_name.replace(".", "_")}.png'), fig)
        else:
            logger.info(
                f"Skipping heatmap for {layer_name}, unsupported weight dimensions: {latest_weights_snapshot.ndim}")

        if fig is None:
            logger.debug(f"Fig was not created for heatmap of {layer_name}")

    def plot_history(self, percent_history: Dict[str, List[Tuple[int, float]]], save_path: str, csv_path: str,
                     xlabel: str = "Global Step"):
        # percent_history is Dict[layer_name, List of (global_step, percentage) tuples]
        logger.info(f"DeadNeuronPlotter.plot_history called. Attempting to plot {len(percent_history)} layers.")
        if not percent_history:
            logger.warning("No dead neuron history (percent_history) collected, skipping plot generation.")
            return

        records = []
        for layer, history_tuples in percent_history.items():
            if not history_tuples:
                logger.debug(f"Layer '{layer}' has empty history in percent_history. Skipping.")
                continue
            for step_val, percentage in history_tuples:
                records.append({"step": step_val, "layer": layer, "percentage": percentage})

        if not records:
            logger.warning("No valid records constructed from percent_history, skipping plot generation.")
            return

        df = pd.DataFrame(records)
        logger.debug(f"DeadNeuronPlotter.plot_history: DataFrame for plotting (head):\n{df.head().to_string()}")

        df.to_csv(csv_path, index=False)
        logger.info(f"Dead neuron percentage history (raw data) saved to {csv_path}")

        unique_layers_in_df = df['layer'].unique()
        if not unique_layers_in_df.size:
            logger.warning("No unique layers found in DataFrame for dead neuron percentage plot. Skipping plot.")
            return

        # Convert 'percentage' to numeric, coercing errors, then drop NaNs before groupby
        df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
        df.dropna(subset=['percentage'], inplace=True)
        if df.empty:
            logger.warning(
                "DataFrame became empty after coercing 'percentage' to numeric and dropping NaNs. Cannot select top layers or plot.")
            return

        if len(unique_layers_in_df) > self.top_n_layers:
            # Group by layer and get the max percentage for each.
            # Ensure that only layers present in the (potentially filtered) df are considered.
            layer_max_percentages = df.groupby("layer")["percentage"].max()
            if layer_max_percentages.empty:
                logger.warning("No layer max percentages to sort for top_n. Plotting all available.")
                top_layers = df['layer'].unique().tolist()  # Fallback to unique layers from potentially filtered df
            else:
                top_layers = layer_max_percentages.sort_values(ascending=False).head(self.top_n_layers).index.tolist()
            logger.info(
                f"Plotting top {len(top_layers)} (max {self.top_n_layers}) layers for dead neuron percentage: {top_layers}")
        else:
            top_layers = unique_layers_in_df.tolist()  # These are layers that had some data
            logger.info(f"Plotting all {len(top_layers)} tracked layers for dead neuron percentage: {top_layers}")

        if not top_layers:
            logger.warning("No layers selected to plot after filtering top_n. Skipping plot.")
            return

        fig_hist, ax_hist = plt.subplots(figsize=(17, 8))
        plotted_anything_history = False

        for layer in top_layers:
            layer_data = df[df["layer"] == layer].sort_values(by="step")
            if not layer_data.empty:
                ax_hist.plot(layer_data['step'], layer_data['percentage'], label=layer, marker='.', linestyle='-')
                plotted_anything_history = True
                logger.debug(
                    f"Plotting data for layer '{layer}'. Steps: {layer_data['step'].tolist()}, Percentages: {layer_data['percentage'].tolist()}")
            else:
                logger.debug(f"No data for layer '{layer}' after filtering for plotting. Skipping this layer.")

        if not plotted_anything_history:
            logger.warning(
                "No data was actually plotted for any layer in dead neuron history. Saving an empty plot might occur or skipping save.")
            plt.close(fig_hist)  # Close the empty figure
            return  # Don't try to save an empty plot

        ax_hist.set_xlabel(xlabel)
        ax_hist.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer=True, min_n_ticks=5))
        plt.xticks(rotation=30, ha='right')
        ax_hist.set_ylabel(f"% of weights < {self.threshold:.1e}")
        ax_hist.set_title("Dead Neuron Weights Percentage Over Time (Tracked Parameters)")

        ax_hist.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize='small')
        ax_hist.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.83, 1])
        self.save_and_close(save_path, fig_hist)

    def save_and_close(self, save_path: str, fig):
        try:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}")
        finally:
            plt.close(fig)

    def plot_all(self, percent_history: Dict[str, List[Tuple[int, float]]],
                 weights_history: Dict[str, List[np.ndarray]]):
        plot_path = os.path.join(self.output_dir, "dead_neuron_percentage_history.png")
        csv_path = os.path.join(self.output_dir, "dead_neuron_percentage_history.csv")
        self.plot_history(percent_history, plot_path, csv_path)

        if not weights_history:
            logger.info("No raw weights history provided to DeadNeuronPlotter. Skipping heatmaps.")
        else:
            for layer_name in weights_history.keys():
                self.plot_heatmap(weights_history, layer_name)


# ActivityPlotter class remains the same as in plotting_utils_diag_v2
class ActivityPlotter:
    """Handles plotting of activation statistics from ActivityMonitor CSV."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir if output_dir else "."
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ActivityPlotter initialized. Plots will be saved to: {self.output_dir}")

    def save_and_close(self, save_path: str, fig):
        try:
            fig.savefig(save_path, bbox_inches='tight'); logger.info(f"Activity plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save activity plot to {save_path}: {e}")
        finally:
            plt.close(fig)

    def plot_activation_stats_evolution(
            self, csv_path: str, target_metric_substring: str = "mean_abs_activation_per_channel",
            target_metric_type: str = "per_channel_overall_mean",
            layers_to_include: Optional[List[str]] = None, max_layers_to_plot: int = 15):
        logger.info(
            f"Plotting activation evolution from: {csv_path} for metric '{target_metric_substring}' type '{target_metric_type}'")
        if not os.path.exists(csv_path): logger.error(f"CSV not found: {csv_path}. Skipping."); return
        try:
            df = pd.read_csv(csv_path)
            if df.empty: logger.warning(f"CSV {csv_path} is empty. Skipping."); return
            if not {'original_metric_name', 'metric_type', 'metric_value', 'global_step', 'layer_identifier'}.issubset(
                    df.columns):
                logger.error(f"CSV {csv_path} missing required columns. Has: {df.columns.tolist()}. Skipping.");
                return
        except Exception as e:
            logger.error(f"Failed to read CSV {csv_path}: {e}. Skipping."); return

        plot_df = df[
            (df['original_metric_name'].astype(str).str.contains(target_metric_substring, case=False, na=False)) &
            (df['metric_type'].astype(str) == target_metric_type)].copy()
        logger.info(f"Found {len(plot_df)} rows matching criteria.")
        if plot_df.empty: logger.warning(f"No data matched. Cannot plot."); return
        plot_df['metric_value'] = pd.to_numeric(plot_df['metric_value'], errors='coerce')
        plot_df.dropna(subset=['metric_value'], inplace=True)
        if plot_df.empty: logger.warning(f"No numeric 'metric_value' data. Cannot plot."); return

        unique_layers = plot_df['layer_identifier'].unique().tolist()
        layers_to_plot_final = unique_layers
        if layers_to_include:
            filtered_by_include = [l for l in unique_layers if any(sub in l for sub in layers_to_include)]
            if filtered_by_include:
                layers_to_plot_final = filtered_by_include
            else:
                logger.warning(f"layers_to_include {layers_to_include} matched no layers.")

        if len(layers_to_plot_final) > max_layers_to_plot:
            layer_max_values = \
            plot_df[plot_df['layer_identifier'].isin(layers_to_plot_final)].groupby('layer_identifier')[
                'metric_value'].max()
            layers_to_plot_final = layer_max_values.nlargest(max_layers_to_plot).index.tolist()
        if not layers_to_plot_final: logger.warning("No layers to plot. Skipping."); return

        fig, ax = plt.subplots(figsize=(17, 8))
        plotted_anything = False
        for layer_id in layers_to_plot_final:
            layer_data = plot_df[plot_df['layer_identifier'] == layer_id].sort_values(by='global_step')
            if not layer_data.empty:
                ax.plot(layer_data['global_step'], layer_data['metric_value'], label=layer_id, marker='.',
                        linestyle='-')
                plotted_anything = True
        if not plotted_anything: logger.warning(f"No data points plotted. Skipping save."); plt.close(fig); return
        ax.set_xlabel("Global Step");
        ax.set_ylabel(f"Value: '{target_metric_substring}' ({target_metric_type})")
        ax.set_title(f"Evolution of '{target_metric_substring}' ({target_metric_type})")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer=True, min_n_ticks=5))
        plt.xticks(rotation=30, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6);
        plt.tight_layout(rect=[0, 0, 0.83, 1])
        filename = f"activation_evo_{target_metric_substring.split('_')[0]}_{target_metric_type.split('_')[-1]}.png"  # Simpler name
        self.save_and_close(os.path.join(self.output_dir, filename.replace(" ", "_").lower()), fig)


def plot_dead_vs_nudge(
    csv_path: str,
    out_png: str,
    nudge_factor: float = 1.05,
    bar_scale: float = 0.5,
):
    """
    Overlays the inactive-channel curve with bars indicating how many
    GroupNorm scales were nudged at each intervention step.

    Parameters
    ----------
    csv_path      CSV written during training with columns:
                     step,inactive_channels,nudged_scales
    out_png       Where to save the PNG.
    nudge_factor  The factor used in 'gentle_nudge_groupnorm_scale';
                  shown in the title for quick reference.
    bar_scale     Multiplier to shrink bar height so it doesn’t hide the line.
    """
    df = pd.read_csv(csv_path, names=["step", "inactive", "nudged"])

    plt.figure(figsize=(9, 4))
    plt.plot(df.step, df.inactive, label="# inactive channels", linewidth=2)
    plt.bar(
        df.step,
        df.nudged * bar_scale,
        width=1.0,
        alpha=0.25,
        label="# scales nudged ×{:.1f}".format(bar_scale),
    )

    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title(f"Dead-channel decay (nudge_factor = {nudge_factor})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

if __name__ == '__main__':
    # ... (Keep the __main__ test block from plotting_utils_diag_v2 for testing ActivityPlotter)
    # ... (And the updated DeadNeuronPlotter test from deadneuron_py_fix_plotting)
    logging.basicConfig(level=logging.DEBUG)
    print("--- Running DeadNeuronPlotter Example (v3) ---")
    dummy_percent_history_fixed = {
        'module.layer1.weight': [(0, 1.0), (20, 0.8), (40, 0.7), (60, 0.75), (80, 0.6)],
        'module.gn1.weight': [(0, 100.0), (20, 50.0), (40, 25.0), (60, 0.0), (80, 0.0)],
    }
    dummy_weights_history_fixed = {
        'module.layer1.weight': [np.random.rand(8, 3, 3, 3) * 0.01],
        'module.gn1.weight': [np.random.rand(8) * 0.001],
    }
    dnp_output_dir = "./results/test_dead_neuron_plots_v3"
    plotter = DeadNeuronPlotter(top_n_layers=3, threshold=1e-5, output_dir=dnp_output_dir)
    plotter.plot_all(dummy_percent_history_fixed, dummy_weights_history_fixed)
    print(f"DeadNeuronPlotter example plots saved to {dnp_output_dir}")

    print("\n--- Running ActivityPlotter Example (v3) ---")
    # ... (ActivityPlotter test from plotting_utils_diag_v2 can be reused or adapted)
