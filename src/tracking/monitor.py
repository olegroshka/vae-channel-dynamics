# src/tracking/monitor.py
import logging
import torch
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ActivityMonitor:
    """
    Placeholder class for monitoring neuron/channel activity.
    This class will be responsible for attaching hooks to target layers
    and collecting relevant statistics during training.
    """
    def __init__(self, model: torch.nn.Module, target_layers: List[str], config: Dict[str, Any]):
        """
        Initializes the monitor.

        Args:
            model: The model to monitor (e.g., SDXLVAEWrapper).
            target_layers: A list of layer names (strings) within the model to attach hooks to.
                           (e.g., "vae.decoder.conv_out", "vae.encoder.conv_in")
            config: Dictionary containing tracking configuration parameters.
        """
        self.model = model
        self.target_layers = target_layers
        self.config = config
        self.collected_data = {} # Dictionary to store tracked data per layer/step
        self.hooks = [] # To store hook handles for removal

        if self.config.get("enabled", False):
            self._register_hooks()
            logger.info(f"ActivityMonitor initialized for layers: {self.target_layers}")
        else:
            logger.info("ActivityMonitor is disabled in config.")

    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        """Retrieves a layer module from the model using its name."""
        modules = layer_name.split('.')
        current_module = self.model
        for mod_name in modules:
            if hasattr(current_module, mod_name):
                current_module = getattr(current_module, mod_name)
            else:
                raise AttributeError(f"Model does not have a layer named '{layer_name}'")
        return current_module

    def _forward_hook(self, layer_name: str):
        """Creates a forward hook function for a specific layer."""
        def hook(module, input, output):
            # --- Placeholder Logic ---
            # This is where you would collect data from the output tensor.
            # Example: Calculate mean activation per channel
            if isinstance(output, torch.Tensor) and output.ndim >= 2: # Need at least batch and channel dims
                # Assuming channels are dim 1 (B, C, H, W) or (B, C, L)
                # Calculate mean activation magnitude per channel across batch and spatial/seq dims
                channel_means = output.abs().mean(dim=[0] + list(range(2, output.ndim))) # Mean over all dims except channel
                step_data = {
                    'mean_activation_per_channel': channel_means.detach().cpu().numpy()
                    # Add other stats: variance, sparsity, quantiles, etc.
                }
                # Store data associated with the current step (needs step info passed somehow)
                # self.collected_data[layer_name][current_step] = step_data
                # logger.debug(f"Hook collected data for {layer_name}")
            else:
                # Handle tuple outputs or other types if necessary
                pass
        return hook

    def _register_hooks(self):
        """Registers forward hooks to the target layers."""
        self.remove_hooks() # Ensure no old hooks are present
        self.collected_data = {name: {} for name in self.target_layers}
        for layer_name in self.target_layers:
            try:
                layer = self._get_layer(layer_name)
                handle = layer.register_forward_hook(self._forward_hook(layer_name))
                self.hooks.append(handle)
                logger.info(f"Registered forward hook for layer: {layer_name}")
            except AttributeError as e:
                logger.error(f"Could not register hook: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error registering hook for {layer_name}: {e}")


    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        # logger.debug("Removed all tracking hooks.")

    def step(self, global_step: int):
        """
        Called at each relevant training step to potentially process collected data.
        (Currently a placeholder - hook logic needs refinement to store step-wise data)
        """
        if not self.config.get("enabled", False):
            return None # Do nothing if disabled

        track_interval = self.config.get("track_interval", 100)
        if global_step % track_interval == 0:
            logger.info(f"Tracking step {global_step} (Placeholder - actual data collection in hooks)")
            # --- Placeholder for processing/logging collected data ---
            # aggregated_metrics = self.aggregate_metrics(global_step)
            # return aggregated_metrics
            return {} # Return empty dict for now
        return None

    def aggregate_metrics(self, global_step: int) -> Dict[str, Any]:
        """
        Placeholder: Aggregates metrics collected by hooks for logging.
        (Requires hooks to store data indexed by step).
        """
        metrics = {}
        logger.warning("ActivityMonitor.aggregate_metrics is a placeholder.")
        # Example:
        # for layer_name, step_data in self.collected_data.items():
        #     if global_step in step_data:
        #         metrics[f"tracking/{layer_name}/mean_activation"] = step_data[global_step]['mean_activation_per_channel'].mean()
        # Clear collected data for the step? Or keep history? Needs design decision.
        return metrics

    def __del__(self):
        """Ensure hooks are removed when the object is deleted."""
        self.remove_hooks()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example Usage (requires a dummy model)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.ReLU()
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(8, 3, 3, padding=1),
                torch.nn.Sigmoid()
            )
            # Add a nested module to test layer name resolution
            self.nested = torch.nn.Sequential( torch.nn.Linear(10,10) )


    model = DummyModel()
    print("Model Architecture:")
    print(model)

    # Example tracking config
    tracking_config = {
        "enabled": True,
        "target_layers": ["encoder.0", "decoder.0", "nested.0", "nonexistent_layer"], # Include a bad name
        "track_interval": 10
    }

    logger.info("\n--- Initializing Monitor ---")
    monitor = ActivityMonitor(model, tracking_config["target_layers"], tracking_config)

    logger.info("\n--- Running Dummy Forward Pass ---")
    dummy_input = torch.randn(4, 3, 32, 32)
    try:
        output = model(dummy_input) # This triggers the hooks
        logger.info("Dummy forward pass successful.")
    except Exception as e:
        logger.error(f"Error during dummy forward pass: {e}")


    logger.info("\n--- Calling Monitor Step ---")
    metrics_step_10 = monitor.step(global_step=10)
    print(f"Metrics at step 10: {metrics_step_10}")
    metrics_step_15 = monitor.step(global_step=15) # Should return None
    print(f"Metrics at step 15: {metrics_step_15}")

    logger.info("\n--- Removing Hooks ---")
    monitor.remove_hooks()
    print(f"Number of active hooks: {len(monitor.hooks)}")

    logger.info("\n--- Deleting Monitor ---")
    del monitor
    # Hooks should be removed by __del__ if not already removed

