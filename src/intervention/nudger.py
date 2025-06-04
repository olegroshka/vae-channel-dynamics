# src/intervention/nudger.py
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class InterventionHandler:
    """
    Applies interventions ("nudges") to model parameters, primarily GroupNorm scales,
    based on classification results identifying inactive channels.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initializes the intervention handler.

        Args:
            model: The PyTorch model (preferably unwrapped if direct parameter
                   modification is done) whose parameters might be modified.
            config: Dictionary containing intervention configuration parameters.
                    Expected keys:
                    - enabled (bool)
                    - strategy (str): e.g., "gentle_nudge_groupnorm_scale", "reset_groupnorm_scale"
                    - nudge_factor (float): Multiplicative factor for gentle nudge.
                    - nudge_value (float): Additive value for gentle nudge (alternative).
                    - max_scale_value (float): Cap for nudged scale parameters.
                    - intervention_interval (int): How often to intervene.
                    - target_layer_param_map (Dict[str, str]): Optional, if classifier provides
                                                               abstract layer names and we need to
                                                               map them to actual param names.
                                                               For now, assumes classifier gives param names.
        """
        self.model = model  # Should be the unwrapped model for direct param access
        self.config = config
        self.strategy = config.get("strategy", "none")
        self.nudge_factor = float(config.get("nudge_factor", 1.1))
        self.nudge_value_add = float(config.get("nudge_value_add", 0.01))
        self.max_scale_value = float(config.get("max_scale_value", 2.0))  # Cap to prevent extreme values
        self.num_nudges_applied = 0

        logger.info(f"InterventionHandler initialized (strategy: {self.strategy}, model type: {type(model)})")
        if not isinstance(model, nn.Module):
            logger.warning(
                f"InterventionHandler received a model of type {type(model)}, expected nn.Module. Parameter access might fail if model is wrapped unexpectedly.")

    def _get_parameter(self, param_name: str) -> Optional[nn.Parameter]:
        """
        Retrieves a parameter from the model using its fully qualified name.
        Assumes self.model is the root model containing the parameter.
        """
        try:
            modules = param_name.split('.')
            current_object = self.model
            for mod_name in modules:
                if hasattr(current_object, mod_name):
                    current_object = getattr(current_object, mod_name)
                else:
                    logger.error(
                        f"Model does not have attribute '{mod_name}' in path '{param_name}'. Current object type: {type(current_object)}")
                    return None

            if isinstance(current_object, nn.Parameter):
                return current_object
            else:
                logger.error(f"Attribute '{param_name}' is not a Parameter, but {type(current_object)}.")
                return None
        except Exception as e:
            logger.error(f"Error getting parameter '{param_name}': {e}", exc_info=True)
            return None

    def intervene(self, classification_results: Dict[str, Any], global_step: int):
        """
        Applies interventions based on classification results.

        Args:
            classification_results: Output from the RegionClassifier.
                                   Example for GroupNorm scale nudge:
                                   {
                                       "vae.decoder.block.0.norm1": { # Key is an identifier for the layer
                                           "param_name_scale": "vae.decoder.block.0.norm1.weight", # Actual param name
                                           "inactive_channel_indices": [0, 5, 12]
                                       }
                                   }
            global_step: The current training step.
        """
        if not self.config.get("enabled", False):
            return
        if self.strategy == "none":
            return

        intervention_interval = self.config.get("intervention_interval", 200)
        if global_step == 0 or global_step % intervention_interval != 0:  # Avoid intervening at step 0 unless interval is 1
            if not (intervention_interval == 1 and global_step > 0):  # Allow if interval is 1 (and not step 0)
                return

        logger.info(
            f"InterventionHandler attempting intervention at step {global_step} with strategy '{self.strategy}'.")
        if not classification_results:
            logger.info(f"Step {global_step}: No regions classified by RegionClassifier, skipping intervention.")
            return

        self.num_nudges_applied = 0

        if self.strategy == "gentle_nudge_groupnorm_scale":
            for layer_key, data in classification_results.items():
                param_name_scale = data.get("param_name_scale")
                inactive_indices = data.get("inactive_channel_indices")

                if not param_name_scale or inactive_indices is None:
                    logger.warning(
                        f"Missing 'param_name_scale' or 'inactive_channel_indices' for layer_key '{layer_key}'. Skipping.")
                    continue

                scale_param = self._get_parameter(param_name_scale)
                if scale_param is None:
                    logger.warning(
                        f"Could not retrieve scale parameter '{param_name_scale}' for layer_key '{layer_key}'. Skipping.")
                    continue

                if not isinstance(scale_param.data, torch.Tensor):
                    logger.warning(f"Parameter data for '{param_name_scale}' is not a tensor. Skipping.")
                    continue

                with torch.no_grad():
                    for idx in inactive_indices:
                        if 0 <= idx < scale_param.data.numel():  # Check bounds (numel for 1D tensor)
                            original_val = scale_param.data[idx].item()
                            # Nudge by multiplying, then ensure it's not too large
                            nudged_val = original_val * self.nudge_factor
                            # Alternative: nudge by adding
                            # nudged_val = original_val + self.nudge_value_add

                            final_val = min(nudged_val, self.max_scale_value)
                            # Optional: ensure a minimum positive value if it was zero
                            # final_val = max(final_val, 1e-6)

                            scale_param.data[idx] = final_val
                            logger.debug(
                                f"Nudged {param_name_scale}[{idx}] from {original_val:.4f} to {final_val:.4f} (pre-cap: {nudged_val:.4f})")
                            self.num_nudges_applied += 1
                        else:
                            logger.warning(
                                f"Inactive index {idx} out of bounds for {param_name_scale} (size: {scale_param.data.numel()})")
            if self.num_nudges_applied > 0:
                logger.info(f"Applied '{self.strategy}' to {self.num_nudges_applied} channel scales at step {global_step}.")

        elif self.strategy == "reset_groupnorm_scale":
            for layer_key, data in classification_results.items():
                param_name_scale = data.get("param_name_scale")
                inactive_indices = data.get("inactive_channel_indices")

                if not param_name_scale or inactive_indices is None:
                    continue

                scale_param = self._get_parameter(param_name_scale)
                if scale_param is None:
                    continue

                with torch.no_grad():
                    for idx in inactive_indices:
                        if 0 <= idx < scale_param.data.numel():
                            original_val = scale_param.data[idx].item()
                            scale_param.data[idx] = 1.0  # Reset to 1.0
                            logger.debug(f"Reset {param_name_scale}[{idx}] from {original_val:.4f} to 1.0")
                            self.num_nudges_applied += 1
            if self.num_nudges_applied > 0:
                logger.info(f"Applied '{self.strategy}' to {self.num_nudges_applied} channel scales at step {global_step}.")
        else:
            logger.warning(f"Unknown intervention strategy: {self.strategy}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Use DEBUG for more verbose output from nudger


    class DummyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Identity()  # Placeholder
            self.decoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.GroupNorm(4, 16),  # num_groups, num_channels
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1)
            )


    class DummyModelWithVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = DummyVAE()


    # This model structure is what InterventionHandler will see
    # The parameter names will be like "vae.decoder.1.weight" (for GroupNorm scale)
    model = DummyModelWithVAE()
    logger.info("Model Structure:")
    # print(model) # verbose
    logger.info("\nParameter Names and Initial GroupNorm Scales:")
    gn_scale_param_name = None
    gn_bias_param_name = None
    for name, param in model.named_parameters():
        # Find the GroupNorm scale parameter name dynamically
        if "decoder.1.weight" in name:  # Assuming decoder.1 is the GroupNorm
            gn_scale_param_name = name
            logger.info(f"- {name}: {param.shape}, Initial values: {param.data[:5].tolist()}")
        if "decoder.1.bias" in name:
            gn_bias_param_name = name

    if gn_scale_param_name is None:
        logger.error("Could not find GroupNorm scale parameter in dummy model. Test will likely fail.")
        exit()

    intervention_config_gentle = {
        "enabled": True,
        "strategy": "gentle_nudge_groupnorm_scale",
        "nudge_factor": 1.2,
        "max_scale_value": 1.5,
        "intervention_interval": 1  # Intervene every step for test
    }

    # Simulate classification results
    # The key is a general identifier, 'param_name_scale' is the actual parameter path
    dummy_classification_results = {
        "decoder_groupnorm_layer_1": {
            "param_name_scale": gn_scale_param_name,
            "inactive_channel_indices": [0, 2, 5, 15]  # Indices of channels in the GroupNorm layer
        }
    }

    logger.info("\n--- Initializing Intervention Handler (Gentle Nudge) ---")
    handler_gentle = InterventionHandler(model, intervention_config_gentle)

    initial_scales = {}
    if gn_scale_param_name:
        param_to_check = handler_gentle._get_parameter(gn_scale_param_name)
        if param_to_check is not None:
            for idx in dummy_classification_results["decoder_groupnorm_layer_1"]["inactive_channel_indices"]:
                if 0 <= idx < param_to_check.data.numel():
                    initial_scales[idx] = param_to_check.data[idx].item()
            logger.info(f"Initial scales for targeted channels of '{gn_scale_param_name}': {initial_scales}")

    logger.info("\n--- Applying Gentle Nudge Intervention (Step 1) ---")
    handler_gentle.intervene(dummy_classification_results, global_step=1)

    logger.info("\nScales after gentle nudge:")
    if gn_scale_param_name:
        param_to_check = handler_gentle._get_parameter(gn_scale_param_name)
        if param_to_check is not None:
            for idx in dummy_classification_results["decoder_groupnorm_layer_1"]["inactive_channel_indices"]:
                if 0 <= idx < param_to_check.data.numel():
                    final_val = param_to_check.data[idx].item()
                    expected_val = min(initial_scales.get(idx, 0) * 1.2, 1.5)
                    logger.info(f"  Channel {idx}: {final_val:.4f} (Expected approx: {expected_val:.4f})")
                    assert abs(final_val - expected_val) < 1e-5, f"Nudge for channel {idx} failed!"

    intervention_config_reset = {
        "enabled": True,
        "strategy": "reset_groupnorm_scale",
        "intervention_interval": 1
    }
    logger.info("\n--- Initializing Intervention Handler (Reset Scale) ---")
    # Re-initialize model to reset scales for this test part
    model_for_reset = DummyModelWithVAE()
    if "decoder.1.weight" in dict(model_for_reset.named_parameters()):  # Ensure param exists
        gn_scale_param_for_reset = "vae.decoder.1.weight"  # Assuming this is the name
        # Set some scales to zero to test reset
        with torch.no_grad():
            model_for_reset.vae.decoder[1].weight.data[0] = 0.0
            model_for_reset.vae.decoder[1].weight.data[2] = 0.1
    else:
        logger.error("GroupNorm scale param not found in new model_for_reset. Skipping reset test.")
        gn_scale_param_for_reset = None

    handler_reset = InterventionHandler(model_for_reset, intervention_config_reset)
    dummy_classification_reset = {
        "decoder_groupnorm_layer_1": {
            "param_name_scale": gn_scale_param_for_reset,
            "inactive_channel_indices": [0, 2]
        }
    }

    logger.info("\n--- Applying Reset Scale Intervention (Step 1) ---")
    if gn_scale_param_for_reset:
        handler_reset.intervene(dummy_classification_reset, global_step=1)

        logger.info("\nScales after reset:")
        param_to_check_reset = handler_reset._get_parameter(gn_scale_param_for_reset)
        if param_to_check_reset is not None:
            for idx in dummy_classification_reset["decoder_groupnorm_layer_1"]["inactive_channel_indices"]:
                if 0 <= idx < param_to_check_reset.data.numel():
                    final_val = param_to_check_reset.data[idx].item()
                    logger.info(f"  Channel {idx}: {final_val:.4f} (Expected: 1.0)")
                    assert abs(final_val - 1.0) < 1e-5, f"Reset for channel {idx} failed!"
    else:
        logger.info("Skipped reset scale test as GroupNorm parameter was not found.")

    logger.info("\n--- Test: Intervention on non-interval step ---")
    intervention_config_gentle["intervention_interval"] = 10
    handler_gentle_interval = InterventionHandler(model, intervention_config_gentle)  # Use original model
    handler_gentle_interval.intervene(dummy_classification_results, global_step=5)  # Should not intervene
    # (No easy assert here without checking logs, but logger should indicate skipping)
