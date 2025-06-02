# src/classification/classifier.py
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch  # Added for type hinting nn.Module

logger = logging.getLogger(__name__)


class RegionClassifier:
    """
    Classifies inactive regions based on tracked data, primarily targeting
    GroupNorm layers for scale parameter nudging.
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):  # Added model parameter
        """
        Initializes the classifier.

        Args:
            model: The unwrapped PyTorch model (e.g., the VAE itself) to inspect for GroupNorm layers.
            config: Dictionary containing classification configuration parameters.
                    Expected keys:
                    - enabled (bool)
                    - method (str): e.g., "threshold_groupnorm_activity"
                    - threshold (float): Value for thresholding.
                    - target_metric_key (str): The metric from ActivityMonitor to use.
                    - layers_to_classify (List[str]): Optional list of specific layer_identifiers
                                                      (from monitor, e.g. "vae.module.norm.output")
                                                      to consider. If empty, considers all.
        """
        self.config = config
        self.method = config.get("method", "threshold_groupnorm_activity")
        self.threshold = float(config.get("threshold", 1e-3))
        self.target_metric_key = config.get("target_metric_key", "mean_abs_activation_per_channel")
        self.layers_to_classify = config.get("layers_to_classify", [])

        self._layer_to_param_map: Dict[
            str, Tuple[str, int]] = {}  # Stores: {monitor_identifier: (param_name, num_channels)}

        if model is not None:
            self._build_groupnorm_map(model)
        else:
            logger.warning("RegionClassifier initialized without a model. Parameter mapping will be heuristic.")

        logger.info(
            f"RegionClassifier initialized (method: {self.method}, threshold: {self.threshold}, "
            f"target_metric: {self.target_metric_key}, Map size: {len(self._layer_to_param_map)})"
        )
        if not self._layer_to_param_map and model is not None:
            logger.warning("RegionClassifier: _layer_to_param_map is empty after _build_groupnorm_map. "
                           "Ensure model has GroupNorm layers and they are accessible.")

    def _build_groupnorm_map(self, model: torch.nn.Module):
        """
        Builds a map from potential ActivityMonitor layer identifiers (output of GroupNorm)
        to their actual scale parameter names and number of channels.
        Example: "vae.encoder.down_blocks.0.resnets.0.norm1.output" -> ("vae.encoder.down_blocks.0.resnets.0.norm1.weight", 320)
        """
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.GroupNorm):
                # ActivityMonitor typically hooks the output of the module.
                monitor_identifier = f"{module_name}.output"
                param_name_scale = f"{module_name}.weight"  # GroupNorm scale parameter
                num_channels = module.num_channels

                # Check if the parameter actually exists
                try:
                    param_obj = model
                    for part in param_name_scale.split('.'):
                        param_obj = getattr(param_obj, part)
                    if not isinstance(param_obj, torch.nn.Parameter):
                        logger.warning(
                            f"Identified GroupNorm '{module_name}', but '{param_name_scale}' is not a Parameter. Skipping.")
                        continue
                except AttributeError:
                    logger.warning(
                        f"Identified GroupNorm '{module_name}', but could not resolve parameter path '{param_name_scale}'. Skipping.")
                    continue

                self._layer_to_param_map[monitor_identifier] = (param_name_scale, num_channels)
                logger.debug(
                    f"RegionClassifier: Mapped GroupNorm output '{monitor_identifier}' to param '{param_name_scale}' (Channels: {num_channels})")
        if not self._layer_to_param_map:
            logger.warning("RegionClassifier: No GroupNorm layers found or mapped in the provided model.")

    def _get_target_param_info_from_map(self, layer_identifier: str) -> Optional[Tuple[str, int]]:
        """
        Uses the pre-built map to get the GroupNorm scale parameter name and num_channels.
        """
        return self._layer_to_param_map.get(layer_identifier)

    def classify(self, tracked_data_for_step: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        """
        Classifies regions based on the input tracked data for the current step.
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
                # layer_identifier is what ActivityMonitor uses, e.g., "vae.encoder.norm1.output"

                # If specific layers are targeted for classification, check if current layer is one of them
                if self.layers_to_classify and layer_identifier not in self.layers_to_classify:
                    logger.debug(f"Step {global_step}: Layer '{layer_identifier}' not in layers_to_classify. Skipping.")
                    continue

                per_channel_values = layer_metrics.get(self.target_metric_key)

                if per_channel_values is None:
                    logger.debug(
                        f"Metric '{self.target_metric_key}' not found for layer '{layer_identifier}' at step {global_step}. Skipping.")
                    continue

                if not isinstance(per_channel_values, np.ndarray) or per_channel_values.ndim != 1:
                    logger.warning(
                        f"Metric '{self.target_metric_key}' for '{layer_identifier}' is not a 1D numpy array. "
                        f"Shape: {per_channel_values.shape if hasattr(per_channel_values, 'shape') else 'N/A'}. Skipping.")
                    continue

                # Attempt to get parameter info using the pre-built map
                param_info = self._get_target_param_info_from_map(layer_identifier)

                if param_info is None:
                    logger.debug(
                        f"Step {global_step}: Layer '{layer_identifier}' not found in pre-built GroupNorm map or not a GN output. Skipping classification for it.")
                    continue

                target_param_name, num_channels_in_gn = param_info

                if per_channel_values.shape[0] != num_channels_in_gn:
                    logger.warning(
                        f"Step {global_step}: Mismatch channels for '{layer_identifier}'. "
                        f"Metric has {per_channel_values.shape[0]}, GroupNorm '{target_param_name}' has {num_channels_in_gn}. Skipping."
                    )
                    continue

                inactive_indices = np.where(per_channel_values < self.threshold)[0]

                if inactive_indices.size > 0:
                    classifier_key = layer_identifier  # Use monitor's identifier as the key

                    classification_results[classifier_key] = {
                        "param_name_scale": target_param_name,
                        "inactive_channel_indices": inactive_indices.tolist(),
                        "metric_used": self.target_metric_key,
                        "threshold_value": self.threshold,
                        "values_of_inactive_channels": per_channel_values[inactive_indices].tolist()
                    }
                    logger.info(
                        f"Step {global_step}: Classified {len(inactive_indices)} inactive channels in '{layer_identifier}' "
                        f"(targeting param '{target_param_name}') using metric '{self.target_metric_key}' < {self.threshold:.2e}.")
                    logger.debug(
                        f"Inactive channel values for {layer_identifier}: {per_channel_values[inactive_indices].tolist()}")
                else:
                    logger.debug(
                        f"Step {global_step}: No inactive channels found for '{layer_identifier}' with threshold {self.threshold:.2e}.")


        elif self.method == "info_geom":
            logger.warning("Information Geometry classification method not implemented.")
            pass
        else:
            logger.warning(f"Unknown classification method: {self.method}")

        return classification_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


    # --- Dummy Model for Testing Classifier's Map Building ---
    class SimpleVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_conv = torch.nn.Conv2d(3, 10, 3, padding=1)
            self.encoder_norm1 = torch.nn.GroupNorm(2, 10)  # 10 channels, 2 groups
            self.encoder_relu = torch.nn.ReLU()
            self.encoder_norm2 = torch.nn.GroupNorm(5, 10)  # Another GN

            self.decoder_norm1 = torch.nn.GroupNorm(4, 20)  # 20 channels
            self.decoder_conv = torch.nn.Conv2d(20, 3, 3, padding=1)


    class ModelWithVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = SimpleVAE()


    test_model = ModelWithVAE()
    logger.info("--- Building Test Model for Classifier ---")
    # Print model structure to verify names
    # for name, module in test_model.named_modules():
    #     if isinstance(module, torch.nn.GroupNorm):
    #         logger.debug(f"Found GroupNorm: {name}, Channels: {module.num_channels}")
    # --- End Dummy Model ---

    classification_config = {
        "enabled": True,
        "method": "threshold_groupnorm_activity",
        "threshold": 0.05,
        "target_metric_key": "mean_abs_activation_per_channel",
        "layers_to_classify": [
            "vae.encoder_norm1.output",  # This is a key from ActivityMonitor
            "vae.decoder_norm1.output"
        ]
    }
    # Initialize classifier WITH the model
    classifier = RegionClassifier(model=test_model, config=classification_config)

    logger.info("\n--- Testing Classifier's Pre-built Map ---")
    expected_map_keys = ["vae.encoder_norm1.output", "vae.encoder_norm2.output", "vae.decoder_norm1.output"]
    for key in expected_map_keys:
        assert key in classifier._layer_to_param_map, f"Expected '{key}' in _layer_to_param_map"

    assert classifier._layer_to_param_map["vae.encoder_norm1.output"] == ("vae.encoder_norm1.weight", 10)
    assert classifier._layer_to_param_map["vae.decoder_norm1.output"] == ("vae.decoder_norm1.weight", 20)
    logger.info("Classifier's _layer_to_param_map seems correctly built for dummy model.")

    dummy_tracked_data_step = {
        # This identifier matches what ActivityMonitor would create for vae.encoder_norm1 output
        "vae.encoder_norm1.output": {
            "mean_abs_activation_per_channel": np.array([0.08, 0.03, 0.6, 0.005, 0.12, 0.01, 0.02, 0.7, 0.04, 0.033]),
            # 10 channels
        },
        # This one should be classified even if not in "layers_to_classify" if that list is empty
        "vae.encoder_norm2.output": {
            "mean_abs_activation_per_channel": np.array([0.1, 0.01, 0.5, 0.02, 0.1, 0.2, 0.03, 0.001, 0.6, 0.04]),
            # 10 channels
        },
        "vae.decoder_norm1.output": {
            "mean_abs_activation_per_channel": np.array([0.01] * 5 + [0.9] * 5 + [0.04] * 5 + [0.002] * 5)
            # 20 channels
        },
        "vae.encoder_conv.output": {  # Not a GroupNorm output, should be skipped by classifier
            "mean_abs_activation_per_channel": np.array([0.1, 0.02, 0.5, 0.01])
        }
    }

    logger.info("\n--- Testing Classifier.classify() Method ---")
    results = classifier.classify(dummy_tracked_data_step, global_step=100)

    print("\n--- Classification Results: ---")
    if results:
        for key, data in results.items():
            print(f"Layer Key (from monitor): {key}")  # This is the ActivityMonitor key
            print(f"  Target Param for Nudge: {data['param_name_scale']}")
            print(f"  Inactive Channel Indices: {data['inactive_channel_indices']}")
            print(f"  Values of Inactive Channels: {data['values_of_inactive_channels']}")
    else:
        print("No inactive channels classified.")

    # --- Assertions for the test case ---
    assert "vae.encoder_norm1.output" in results, "Expected 'vae.encoder_norm1.output' to be in results"
    if "vae.encoder_norm1.output" in results:
        res_enc_norm1 = results["vae.encoder_norm1.output"]
        assert res_enc_norm1["param_name_scale"] == "vae.encoder_norm1.weight"
        # Channels with values < 0.05: 0.03 (idx 1), 0.005 (idx 3), 0.01 (idx 5), 0.02 (idx 6), 0.04 (idx 8), 0.033 (idx 9)
        expected_indices = [1, 3, 5, 6, 8, 9]
        assert all(idx in res_enc_norm1["inactive_channel_indices"] for idx in expected_indices)
        assert len(res_enc_norm1["inactive_channel_indices"]) == len(expected_indices)

    # vae.encoder_norm2.output was NOT in layers_to_classify, so it should NOT be in results
    assert "vae.encoder_norm2.output" not in results, \
        "'vae.encoder_norm2.output' should NOT be in results as it's not in 'layers_to_classify'"

    assert "vae.decoder_norm1.output" in results, "Expected 'vae.decoder_norm1.output' to be in results"
    if "vae.decoder_norm1.output" in results:
        res_dec_norm1 = results["vae.decoder_norm1.output"]
        assert res_dec_norm1["param_name_scale"] == "vae.decoder_norm1.weight"
        # Channels < 0.05: first 5 (0.01), last 5 (0.04), last 5 (0.002)
        # Indices: 0,1,2,3,4 (for 0.01), 10,11,12,13,14 (for 0.04), 15,16,17,18,19 (for 0.002)
        expected_indices_dec = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert all(idx in res_dec_norm1["inactive_channel_indices"] for idx in expected_indices_dec)
        assert len(res_dec_norm1["inactive_channel_indices"]) == len(expected_indices_dec)

    assert "vae.encoder_conv.output" not in results, \
        "Non-GroupNorm layer 'encoder_conv.output' should not be in results as it won't be in the map."

    # Test with empty layers_to_classify (should process all applicable from map)
    logger.info("\n--- Testing Classifier with empty 'layers_to_classify' ---")
    classification_config_all_layers = {
        "enabled": True, "method": "threshold_groupnorm_activity", "threshold": 0.05,
        "target_metric_key": "mean_abs_activation_per_channel",
        "layers_to_classify": []  # Empty list
    }
    classifier_all_layers = RegionClassifier(model=test_model, config=classification_config_all_layers)
    results_all_layers = classifier_all_layers.classify(dummy_tracked_data_step, global_step=101)

    assert "vae.encoder_norm1.output" in results_all_layers
    assert "vae.encoder_norm2.output" in results_all_layers  # Should be present now
    assert "vae.decoder_norm1.output" in results_all_layers
    logger.info("Classifier test with empty 'layers_to_classify' completed successfully.")

    logger.info("\nClassifier tests completed successfully.")