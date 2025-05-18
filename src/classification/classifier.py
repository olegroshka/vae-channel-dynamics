# src/classification/classifier.py
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class RegionClassifier:
    """
    Classifies inactive regions based on tracked data, primarily targeting
    GroupNorm layers for scale parameter nudging.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the classifier.

        Args:
            config: Dictionary containing classification configuration parameters.
                    Expected keys:
                    - enabled (bool)
                    - method (str): e.g., "threshold_groupnorm_activity"
                    - threshold (float): Value for thresholding.
                    - target_metric_key (str): The metric from ActivityMonitor to use for classification
                                               (e.g., "mean_abs_activation_per_channel").
                    - layers_to_classify (List[str]): Optional list of specific layer_identifiers
                                                      (from monitor, e.g. "vae.module.norm.output")
                                                      to consider. If empty, considers all.
        """
        self.config = config
        self.method = config.get("method", "threshold_groupnorm_activity")
        self.threshold = float(config.get("threshold", 1e-3))  # Ensure float
        self.target_metric_key = config.get("target_metric_key", "mean_abs_activation_per_channel")
        self.layers_to_classify = config.get("layers_to_classify", [])  # If empty, process all available

        logger.info(
            f"RegionClassifier initialized (method: {self.method}, threshold: {self.threshold}, target_metric: {self.target_metric_key})")

    def _map_layer_identifier_to_groupnorm_param(self, layer_identifier: str) -> Optional[str]:
        """
        Maps a layer identifier (from ActivityMonitor, e.g., "vae.module.norm1.output")
        to a potential GroupNorm scale parameter name (e.g., "vae.module.norm1.weight").

        This is a heuristic and assumes:
        1. The layer_identifier ends with ".input" or ".output" if it's from a hook.
        2. The underlying module is a GroupNorm and its scale parameter is '.weight'.
        3. The identifier parts directly map to module hierarchy.

        Args:
            layer_identifier: The identifier string from ActivityMonitor.

        Returns:
            The potential parameter name string or None if mapping fails.
        """
        if not isinstance(layer_identifier, str):
            return None

        # Heuristic: Assume GroupNorm layers are often named 'norm', 'gn', 'group_norm'
        # and we are interested if the identifier points to such a layer.
        # For this classifier, we are typically interested in the *output* activations of a norm layer
        # to decide if its *scale parameters* (weights) are causing suppression.

        base_layer_name = layer_identifier
        if base_layer_name.endswith(".output") or base_layer_name.endswith(".input"):
            base_layer_name = base_layer_name.rsplit('.', 1)[0]

        # Only proceed if it looks like a norm layer that would have a .weight for scale
        # This is a simple check; more robust would be to inspect model structure.
        if "norm" in base_layer_name.lower():  # A common naming convention
            return f"{base_layer_name}.weight"  # GroupNorm scale parameter is typically '.weight'

        logger.debug(
            f"Layer identifier '{layer_identifier}' does not seem to be a targetable GroupNorm for scale nudging based on name. Skipping parameter mapping.")
        return None

    def classify(self, tracked_data_for_step: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        """
        Classifies regions based on the input tracked data for the current step.

        Args:
            tracked_data_for_step: Data collected by ActivityMonitor for the current step.
                                   Structure: {layer_identifier: {metric_name: value_array}}
                                   e.g., {"vae.encoder.norm1.output": {"mean_abs_activation_per_channel": np.array([...])}}
            global_step: The current training step.

        Returns:
            A dictionary structured for InterventionHandler:
            {
                "classifier_key_for_layer_X": {
                    "param_name_scale": "actual.path.to.groupnorm.weight.parameter",
                    "inactive_channel_indices": [idx1, idx2, ...],
                    "metric_used": "mean_abs_activation_per_channel",
                    "threshold_value": self.threshold,
                    "values_of_inactive_channels": [val1, val2, ...]
                },
                ...
            }
        """
        if not self.config.get("enabled", False):
            return {}

        classification_results = {}
        logger.debug(f"Classifier running at step {global_step} with method {self.method}.")

        if self.method == "threshold_groupnorm_activity":
            if not tracked_data_for_step:
                logger.debug(f"Step {global_step}: No tracked data provided to classifier.")
                return {}

            for layer_identifier, layer_metrics in tracked_data_for_step.items():
                # If specific layers are targeted for classification, check if current layer is one of them
                if self.layers_to_classify and layer_identifier not in self.layers_to_classify:
                    continue

                per_channel_values = layer_metrics.get(self.target_metric_key)

                if per_channel_values is None:
                    logger.debug(
                        f"Metric '{self.target_metric_key}' not found for layer '{layer_identifier}' at step {global_step}. Skipping.")
                    continue

                if not isinstance(per_channel_values, np.ndarray) or per_channel_values.ndim != 1:
                    logger.warning(
                        f"Metric '{self.target_metric_key}' for '{layer_identifier}' is not a 1D numpy array. Shape: {per_channel_values.shape if hasattr(per_channel_values, 'shape') else 'N/A'}. Skipping.")
                    continue

                # Attempt to map this layer_identifier to a GroupNorm scale parameter
                # This classifier specifically targets GroupNorm scale parameters for nudging.
                # We typically look at the *output* of a GroupNorm to see if its channels are suppressed.
                # The parameter to nudge would be the .weight of that GroupNorm layer.
                target_param_name = self._map_layer_identifier_to_groupnorm_param(layer_identifier)

                if target_param_name is None:
                    # This means the layer_identifier (e.g., "conv_in.output") wasn't mapped to a GroupNorm weight.
                    # This is expected for non-GroupNorm layers or if the naming heuristic fails.
                    logger.debug(
                        f"Skipping classification for '{layer_identifier}' as it's not identified as a targetable GroupNorm for scale nudging.")
                    continue

                inactive_indices = np.where(per_channel_values < self.threshold)[0]

                if inactive_indices.size > 0:
                    # Use a descriptive key for classification_results, can be layer_identifier itself
                    # or a more abstract key if needed later.
                    classifier_key = layer_identifier  # Or a more abstract key if mapping is complex

                    classification_results[classifier_key] = {
                        "param_name_scale": target_param_name,  # Parameter to be nudged
                        "inactive_channel_indices": inactive_indices.tolist(),
                        "metric_used": self.target_metric_key,
                        "threshold_value": self.threshold,
                        "values_of_inactive_channels": per_channel_values[inactive_indices].tolist()
                        # For logging/debug
                    }
                    logger.info(
                        f"Step {global_step}: Classified {len(inactive_indices)} inactive channels in '{layer_identifier}' (targeting param '{target_param_name}') using metric '{self.target_metric_key}' < {self.threshold:.2e}.")
                    logger.debug(
                        f"Inactive channel values for {layer_identifier}: {per_channel_values[inactive_indices].tolist()}")

        elif self.method == "info_geom":
            logger.warning("Information Geometry classification method not implemented.")
            pass
        else:
            logger.warning(f"Unknown classification method: {self.method}")

        return classification_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Use DEBUG for more verbose output

    # Example Config
    classification_config = {
        "enabled": True,
        "method": "threshold_groupnorm_activity",
        "threshold": 0.05,
        "target_metric_key": "mean_abs_activation_per_channel",
        # "layers_to_classify": ["vae.encoder.norm1.output"] # Optional: to focus
    }
    classifier = RegionClassifier(classification_config)

    # Dummy tracked data (simulating output from ActivityMonitor.get_data_for_step())
    # Keys are layer_identifiers (e.g., "module_name.capture_point")
    # Values are dicts of metric_name to numpy array (or tensor if full_activation_map)
    dummy_tracked_data_step = {
        "vae.encoder.conv_in.output": {  # This is not a GroupNorm, so should be skipped by mapping
            "mean_abs_activation_per_channel": np.array([0.1, 0.02, 0.5, 0.01])
        },
        "vae.encoder.norm1.output": {  # This should be identified as a GroupNorm
            "mean_abs_activation_per_channel": np.array([0.08, 0.03, 0.6, 0.005, 0.12]),
            "std_activation": np.array(0.2)  # Other metrics might be present
        },
        "vae.decoder.norm_final.output": {  # Another GroupNorm example
            "mean_abs_activation_per_channel": np.array([0.01, 0.9, 0.04])
        },
        "some_other_layer.output": {  # Not a norm layer
            "mean_abs_activation_per_channel": np.array([0.001, 0.002])
        }
    }

    logger.info("\n--- Testing Classifier ---")
    results = classifier.classify(dummy_tracked_data_step, global_step=100)

    print("\n--- Classification Results: ---")
    if results:
        for key, data in results.items():
            print(f"Layer Key (in results): {key}")
            print(f"  Target Param for Nudge: {data['param_name_scale']}")
            print(f"  Inactive Channel Indices: {data['inactive_channel_indices']}")
            print(f"  Metric Used: {data['metric_used']}")
            print(f"  Threshold: {data['threshold_value']}")
            print(f"  Values of Inactive Channels: {data['values_of_inactive_channels']}")
    else:
        print("No inactive channels classified.")

    # --- Assertions for the test case ---
    assert "vae.encoder.norm1.output" in results, "Expected 'vae.encoder.norm1.output' to be in results"
    if "vae.encoder.norm1.output" in results:
        res_norm1 = results["vae.encoder.norm1.output"]
        assert res_norm1["param_name_scale"] == "vae.encoder.norm1.weight"
        assert 1 in res_norm1["inactive_channel_indices"]  # 0.03 < 0.05
        assert 3 in res_norm1["inactive_channel_indices"]  # 0.005 < 0.05
        assert 0 not in res_norm1["inactive_channel_indices"]  # 0.08 > 0.05
        assert len(res_norm1["inactive_channel_indices"]) == 2

    assert "vae.decoder.norm_final.output" in results, "Expected 'vae.decoder.norm_final.output' to be in results"
    if "vae.decoder.norm_final.output" in results:
        res_norm_final = results["vae.decoder.norm_final.output"]
        assert res_norm_final["param_name_scale"] == "vae.decoder.norm_final.weight"
        assert 0 in res_norm_final["inactive_channel_indices"]  # 0.01 < 0.05
        assert 2 in res_norm_final["inactive_channel_indices"]  # 0.04 < 0.05
        assert len(res_norm_final["inactive_channel_indices"]) == 2

    assert "vae.encoder.conv_in.output" not in results, "Non-GroupNorm layer 'conv_in.output' should not be in results for nudging."
    assert "some_other_layer.output" not in results, "Non-GroupNorm layer 'some_other_layer.output' should not be in results for nudging."

    logger.info("\nClassifier test completed successfully.")

