# src/tracking/deadneuron.py
import logging

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Type  # Corrected Type import
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeadNeuronTracker:

    def __init__(self,
                 target_layer_classes: Tuple[Type[nn.Module], ...],
                 target_layer_names_for_raw_weights: List[str],  # Renamed for clarity
                 threshold: float,
                 mean_percentage: float,
                 dead_type: str = "threshold"):
        """Tracks percentage of 'dead' neurons in specified layers of a model."""
        self.threshold = threshold
        self.mean_percentage = mean_percentage
        self.target_layer_classes = target_layer_classes
        # These are specific full parameter names for raw weight history, e.g., "encoder.conv_in.weight"
        self.target_layer_names_for_raw_weights = target_layer_names_for_raw_weights

        if dead_type == "threshold":
            self.get_percentage = self.smaller_than_threshold
        elif dead_type == "percent_of_mean":
            self.get_percentage = self.percent_of_mean
        elif dead_type == "both":
            self.get_percentage = self.both
        else:
            logger.warning(f"Unknown dead_type: {dead_type}. Defaulting to no-op for percentage calculation.")
            self.get_percentage = self.noop

        self.weights_history = defaultdict(list)  # Stores raw weights for specified named parameters
        self.percent_history = defaultdict(list)  # Stores dead % for layers of target_layer_classes

    def track_dead_neurons(self, model_wrapper: nn.Module):
        """
        Tracks the percentage of near-zero weights in specified layers of the model.
        Args:
            model_wrapper: The model wrapper (e.g., SDXLVAEWrapper) which has a `.vae` attribute
                           pointing to the actual VAE model (e.g., AutoencoderKL), OR it could be
                           the VAE model itself if no wrapper is used in a specific context.
        """
        actual_vae_model = None
        if hasattr(model_wrapper, 'vae') and model_wrapper.vae is not None:
            actual_vae_model = model_wrapper.vae
            logger.debug("DeadNeuronTracker: Using model_wrapper.vae")
        elif isinstance(model_wrapper, nn.Module):  # Check if model_wrapper itself could be the VAE
            # This branch might be hit if a raw AutoencoderKL is passed directly for some reason
            actual_vae_model = model_wrapper
            logger.debug("DeadNeuronTracker: Using model_wrapper directly as the VAE model.")
        else:
            logger.error(
                "DeadNeuronTracker: model_wrapper is not an nn.Module or has no .vae attribute. Cannot track neurons.")
            return

        if actual_vae_model is None:  # Should be caught by above, but as a safeguard
            logger.error("DeadNeuronTracker: actual_vae_model is None. Cannot track neurons.")
            return

        for name, param in actual_vae_model.named_parameters():
            if not param.requires_grad:  # Skip non-trainable parameters
                continue

            # For storing raw weights of specifically named parameters
            if name in self.target_layer_names_for_raw_weights:
                self.weights_history[name].append(param.detach().cpu().numpy())
                logger.debug(f"DeadNeuronTracker: Stored raw weights for '{name}'.")

            # For calculating dead percentage in layers of target_layer_classes
            # We typically track 'weight' parameters of these layers.
            if "weight" in name:  # Heuristic: usually weights are named 'weight'
                module_path = ".".join(name.split(".")[:-1])
                try:
                    module = actual_vae_model.get_submodule(module_path)
                    if isinstance(module, self.target_layer_classes):
                        percentage = self.get_percentage(param)
                        self.percent_history[name].append(percentage)
                        logger.debug(
                            f"DeadNeuronTracker: Tracked {name}, Type: {type(module).__name__}, Dead %: {percentage:.2f}%")
                except AttributeError:
                    logger.debug(
                        f"DeadNeuronTracker: Could not get submodule for {module_path} from param {name}. Skipping % calc.")
                    continue
                except Exception as e:
                    logger.error(f"DeadNeuronTracker: Error processing param {name} for percentage: {e}")

    def noop(self, param):
        return 0.0

    def smaller_than_threshold(self, param: torch.Tensor) -> float:
        total_elements = param.numel()
        if total_elements == 0:
            return 0.0
        small_values = (param.abs() < self.threshold).sum().item()
        return (small_values / total_elements) * 100.0

    def percent_of_mean(self, param: torch.Tensor) -> float:
        if param.numel() == 0: return 0.0
        param_abs = param.abs()
        mean_abs = param_abs.mean().item()

        if abs(mean_abs) < 1e-9:  # Check if mean_abs is effectively zero
            # If mean is zero, any non-zero element is infinitely larger.
            # If all elements are zero, then 100% are "dead" by this metric.
            is_all_zero = (param_abs < 1e-9).all().item()  # Check if all elements are effectively zero
            return 100.0 if is_all_zero else 0.0

        adaptive_threshold = self.mean_percentage * mean_abs
        small_values = (param_abs < adaptive_threshold).sum().item()
        total_elements = param.numel()
        return (small_values / total_elements) * 100.0

    def both(self, param: torch.Tensor) -> float:
        total_elements = param.numel()
        if total_elements == 0:
            return 0.0

        param_abs = param.abs()
        condition_fixed = param_abs < self.threshold

        mean_abs = param_abs.mean().item()
        if abs(mean_abs) < 1e-9:  # Check if mean_abs is effectively zero
            is_all_zero = (param_abs < 1e-9).all().item()
            condition_adaptive = torch.full_like(condition_fixed, fill_value=is_all_zero)
        else:
            adaptive_threshold = self.mean_percentage * mean_abs
            condition_adaptive = param_abs < adaptive_threshold

        combined_condition = condition_fixed & condition_adaptive
        small_values = combined_condition.sum().item()
        return (small_values / total_elements) * 100.0


if __name__ == '__main__':
    # Configure basic logging for testing this module
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to see all tracker logs
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )


    # --- Dummy Model for Testing ---
    class DummyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv layer whose weights we might track percentage for
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.conv1.weight.data.fill_(0.001)  # Some small weights
            self.conv1.weight.data[0, 0, 0, 0] = 1.0  # One large weight
            self.conv1.weight.data[1, 0, 0, 0] = 0.0000001  # One very small weight (below typical threshold)

            # Linear layer
            self.fc1 = nn.Linear(10, 2)
            self.fc1.weight.data.normal_(0, 0.01)  # Small random weights

            # Another Conv layer we might explicitly ask for raw weight history
            self.another_conv = nn.Conv2d(8, 4, kernel_size=1)
            self.another_conv.weight.data.fill_(0.5)

            # A layer not in target_layer_classes or target_layer_names
            self.relu = nn.ReLU()


    # Wrapper class similar to SDXLVAEWrapper for testing structure
    class ModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = DummyVAE()


    test_model_wrapper = ModelWrapper()
    # --- End Dummy Model ---

    logger.info("--- Testing DeadNeuronTracker ---")

    # Define parameters for the tracker
    TARGET_CLASSES = (nn.Conv2d, nn.Linear)  # Track Conv2d and Linear layers for dead %
    # Ask for raw weight history for 'another_conv.weight'
    TARGET_NAMES_RAW = ["another_conv.weight"]
    THRESHOLD_VAL = 1e-5
    MEAN_PERCENT_VAL = 0.1  # 10% of mean

    # Test 1: Dead type "threshold"
    logger.info("\n--- Test 1: dead_type = 'threshold' ---")
    tracker_thresh = DeadNeuronTracker(
        target_layer_classes=TARGET_CLASSES,
        target_layer_names_for_raw_weights=TARGET_NAMES_RAW,
        threshold=THRESHOLD_VAL,
        mean_percentage=MEAN_PERCENT_VAL,  # Not used by 'threshold' type
        dead_type="threshold"
    )
    tracker_thresh.track_dead_neurons(test_model_wrapper)
    print("Percentage History (threshold):")
    for layer_name, history in tracker_thresh.percent_history.items():
        print(f"  {layer_name}: {history}")
    print("Weights History (raw values for targeted names):")
    for layer_name, history_list in tracker_thresh.weights_history.items():
        print(
            f"  {layer_name}: found {len(history_list)} snapshots, first snapshot shape: {history_list[0].shape if history_list else 'N/A'}")

    # Expected for conv1.weight (3*8*3*3 = 216 elements):
    # Most are 0.001. One is 1.0. One is 0.0000001 (this one is < 1e-5).
    # So, 1 out of 216 elements should be counted by threshold. (1/216 * 100) approx 0.46%
    assert "conv1.weight" in tracker_thresh.percent_history
    assert abs(tracker_thresh.percent_history["conv1.weight"][0] - (1 / 216 * 100)) < 0.01

    # Test 2: Dead type "percent_of_mean"
    logger.info("\n--- Test 2: dead_type = 'percent_of_mean' ---")
    # Re-initialize model to reset weights if they were modified (not in this tracker)
    test_model_wrapper_2 = ModelWrapper()
    test_model_wrapper_2.vae.conv1.weight.data.fill_(0.1)  # All weights are 0.1
    test_model_wrapper_2.vae.conv1.weight.data[0, 0, 0, 0] = 0.001  # One weight is 0.001
    # Mean of abs weights will be close to 0.1. 10% of mean is ~0.01.
    # The weight 0.001 is < 0.01. So 1 element should be dead.

    tracker_mean = DeadNeuronTracker(
        target_layer_classes=TARGET_CLASSES,
        target_layer_names_for_raw_weights=TARGET_NAMES_RAW,
        threshold=THRESHOLD_VAL,  # Not used by 'percent_of_mean'
        mean_percentage=MEAN_PERCENT_VAL,
        dead_type="percent_of_mean"
    )
    tracker_mean.track_dead_neurons(test_model_wrapper_2)
    print("Percentage History (percent_of_mean):")
    for layer_name, history in tracker_mean.percent_history.items():
        print(f"  {layer_name}: {history}")
    assert "conv1.weight" in tracker_mean.percent_history
    assert abs(tracker_mean.percent_history["conv1.weight"][0] - (1 / 216 * 100)) < 0.01

    # Test 3: Dead type "both"
    logger.info("\n--- Test 3: dead_type = 'both' ---")
    test_model_wrapper_3 = ModelWrapper()
    # Setup: threshold = 1e-5, mean_percentage = 0.1
    # Weight 1: 0.000001 (value < threshold=1e-5)
    # Weight 2: 0.01 (value > threshold=1e-5)
    # Other weights: 1.0
    # Make param_abs: [1e-6, 0.01, 1.0, 1.0, ..., 1.0]
    # Mean_abs will be close to 1.0. adaptive_threshold = 0.1 * 1.0 = 0.1
    # Weight 1 (1e-6): < threshold (T), < adaptive_threshold (T) => BOTH (T)
    # Weight 2 (0.01): > threshold (F), < adaptive_threshold (T) => BOTH (F)
    test_model_wrapper_3.vae.conv1.weight.data.fill_(1.0)
    test_model_wrapper_3.vae.conv1.weight.data[0, 0, 0, 0] = 0.000001  # Satisfies both
    test_model_wrapper_3.vae.conv1.weight.data[0, 0, 0, 1] = 0.01  # Satisfies only adaptive

    tracker_both = DeadNeuronTracker(
        target_layer_classes=TARGET_CLASSES,
        target_layer_names_for_raw_weights=TARGET_NAMES_RAW,
        threshold=THRESHOLD_VAL,
        mean_percentage=MEAN_PERCENT_VAL,
        dead_type="both"
    )
    tracker_both.track_dead_neurons(test_model_wrapper_3)
    print("Percentage History (both):")
    for layer_name, history in tracker_both.percent_history.items():
        print(f"  {layer_name}: {history}")
    assert "conv1.weight" in tracker_both.percent_history
    assert abs(tracker_both.percent_history["conv1.weight"][0] - (1 / 216 * 100)) < 0.01

    # Test 4: Raw weight tracking
    logger.info("\n--- Test 4: Raw weight history ---")
    assert "another_conv.weight" in tracker_thresh.weights_history  # From first tracker run
    assert len(tracker_thresh.weights_history["another_conv.weight"]) == 1
    assert tracker_thresh.weights_history["another_conv.weight"][0].shape == (4, 8, 1, 1)  # From DummyVAE
    assert np.all(tracker_thresh.weights_history["another_conv.weight"][0] == 0.5)

    logger.info("\n--- DeadNeuronTracker tests completed successfully! ---")