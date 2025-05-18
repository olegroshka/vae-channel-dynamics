# src/tracking/monitor.py
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Callable, Optional  # <<< Added Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ActivityMonitor:
    """
    Monitors neuron/channel activity by attaching hooks to target layers
    and collecting relevant statistics during training.
    Can capture layer inputs (pre-hooks) or outputs (post-hooks).
    """

    def __init__(self, model: torch.nn.Module, tracking_config: Dict[str, Any]):
        """
        Initializes the monitor.

        Args:
            model: The model to monitor (e.g., SDXLVAEWrapper).
            tracking_config: Dictionary containing tracking configuration.
        """
        self.model = model
        self.config = tracking_config
        self.target_layers_config: List[Dict[str, Any]] = self.config.get("target_layers", [])

        self.hook_collected_buffer = defaultdict(lambda: defaultdict(list))
        self.processed_data_by_step = defaultdict(dict)  # Stores {global_step: {layer_id: {metric: val}}}

        self.hooks = []

        if self.config.get("enabled", False):
            self._register_hooks()
            logger.info(f"ActivityMonitor initialized for {len(self.target_layers_config)} target(s).")
        else:
            logger.info("ActivityMonitor is disabled in config.")

    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        modules = layer_name.split('.')
        current_module = self.model
        for mod_name in modules:
            if hasattr(current_module, mod_name):
                current_module = getattr(current_module, mod_name)
            else:
                # If using accelerator, model might be wrapped. Try accessing .module
                if hasattr(current_module, 'module') and hasattr(current_module.module, mod_name):
                    current_module = getattr(current_module.module, mod_name)
                else:
                    raise AttributeError(
                        f"Model (or its .module) does not have a layer named '{layer_name}' (path: {mod_name})")
        return current_module

    def _calculate_metrics(self, tensor: torch.Tensor, metrics_to_capture: List[str]) -> Dict[str, Any]:
        calculated = {}
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"Cannot calculate metrics for non-tensor type: {type(tensor)}")
            return calculated

        for metric_name in metrics_to_capture:
            try:
                if metric_name == 'mean_abs_activation_per_channel':
                    if tensor.ndim >= 2:
                        channel_means = tensor.abs().mean(dim=[0] + list(range(2, tensor.ndim)))
                        calculated[metric_name] = channel_means.detach().cpu().numpy()
                    else:
                        calculated[metric_name] = tensor.abs().mean().detach().cpu().numpy()
                elif metric_name == 'full_activation_map':
                    calculated[metric_name] = tensor.detach().clone().cpu()  # Stored as PyTorch CPU Tensor
                elif metric_name == 'mean_activation':
                    calculated[metric_name] = tensor.mean().detach().cpu().numpy()
                elif metric_name == 'std_activation':
                    calculated[metric_name] = tensor.std().detach().cpu().numpy()
                else:
                    logger.warning(f"Unknown metric '{metric_name}' requested.")
            except Exception as e:
                logger.error(f"Error calculating metric '{metric_name}': {e}", exc_info=True)
        return calculated

    def _create_hook_fn(self, layer_config: Dict[str, Any], capture_point: str) -> Callable:
        layer_name = layer_config['name']
        metrics_to_capture = layer_config.get('metrics', ['mean_abs_activation_per_channel'])
        layer_identifier = f"{layer_name}.{capture_point}"

        def hook_fn(module: torch.nn.Module, hook_input: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                    hook_output: Optional[torch.Tensor] = None):  # Optional is used here
            tensor_to_process = None
            if capture_point == 'input':
                if isinstance(hook_input, tuple) and len(hook_input) > 0:
                    tensor_to_process = hook_input[0]
                elif isinstance(hook_input, torch.Tensor):
                    tensor_to_process = hook_input
            elif capture_point == 'output':
                tensor_to_process = hook_output

            if tensor_to_process is not None and isinstance(tensor_to_process, torch.Tensor):
                calculated_metrics = self._calculate_metrics(tensor_to_process, metrics_to_capture)
                for metric_name, value in calculated_metrics.items():
                    self.hook_collected_buffer[layer_identifier][metric_name].append(value)
            elif tensor_to_process is not None:
                logger.debug(
                    f"Hook for {layer_identifier}: tensor_to_process is not a Tensor (type: {type(tensor_to_process)}). Skipping.")

        return hook_fn

    def _register_hooks(self):
        self.remove_hooks()
        self.hook_collected_buffer.clear()

        for layer_conf in self.target_layers_config:
            layer_name = layer_conf.get("name")
            capture_point = layer_conf.get("capture_point", "output")
            if not layer_name:
                logger.warning("Skipping a target_layer entry with no name.")
                continue

            layer_identifier = f"{layer_name}.{capture_point}"
            logger.debug(f"Attempting to register hook for: {layer_identifier}")

            try:
                layer = self._get_layer(layer_name)
                hook_function = self._create_hook_fn(layer_conf, capture_point)

                if capture_point == 'input':
                    handle = layer.register_forward_pre_hook(hook_function)
                    self.hooks.append(handle)
                    logger.info(f"Registered FORWARD PRE-HOOK for layer: {layer_name} (input)")
                elif capture_point == 'output':
                    handle = layer.register_forward_hook(hook_function)
                    self.hooks.append(handle)
                    logger.info(f"Registered FORWARD HOOK for layer: {layer_name} (output)")
                else:
                    logger.warning(f"Unknown capture_point '{capture_point}' for layer {layer_name}. Skipping.")
            except AttributeError as e:
                logger.error(f"Could not register hook for {layer_identifier} (AttributeError): {e}")
            except Exception as e:
                logger.error(f"Unexpected error registering hook for {layer_identifier}: {e}", exc_info=True)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def step(self, global_step: int) -> Dict[str, Any]:
        if not self.config.get("enabled", False):
            return {}

        track_interval = self.config.get("track_interval", 100)
        if global_step % track_interval != 0:
            return {}

        logger.debug(f"ActivityMonitor processing step {global_step}")
        wandb_metrics = {}
        current_step_processed_data = {}

        for layer_identifier, metric_data in self.hook_collected_buffer.items():
            current_step_processed_data[layer_identifier] = {}
            for metric_name, values_list in metric_data.items():
                if not values_list:
                    continue

                aggregated_value = None
                try:
                    if metric_name == 'full_activation_map':
                        aggregated_value = values_list[0]
                        if isinstance(aggregated_value, torch.Tensor):
                            numpy_value = aggregated_value.numpy()
                            wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_mean"] = np.mean(
                                numpy_value.astype(np.float32))
                            wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_std"] = np.std(
                                numpy_value.astype(np.float32))
                        elif isinstance(aggregated_value, np.ndarray):
                            wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_mean"] = np.mean(
                                aggregated_value.astype(np.float32))
                            wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_std"] = np.std(
                                aggregated_value.astype(np.float32))
                    elif 'mean_abs_activation_per_channel' in metric_name:
                        if all(isinstance(v, np.ndarray) for v in values_list):
                            stacked_values = np.stack(values_list)
                            aggregated_value = np.mean(stacked_values, axis=0)
                            wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_overall_mean"] = np.mean(
                                aggregated_value)
                            wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_overall_std"] = np.std(
                                aggregated_value)
                        else:
                            aggregated_value = values_list[0] if values_list else None
                            if isinstance(aggregated_value, np.ndarray):  # Check if it's an ndarray after fallback
                                wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_overall_mean"] = np.mean(
                                    aggregated_value)
                                wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_overall_std"] = np.std(
                                    aggregated_value)  # Add std for consistency
                            elif aggregated_value is not None:  # If not ndarray but not None (e.g. scalar from fallback)
                                wandb_metrics[f"tracking/{layer_identifier}/{metric_name}_overall_mean"] = float(
                                    aggregated_value)


                    else:
                        scalar_values = [v.item() if hasattr(v, 'item') else float(v) for v in values_list]
                        aggregated_value = np.mean(scalar_values)
                        wandb_metrics[f"tracking/{layer_identifier}/{metric_name}"] = aggregated_value
                except Exception as e:
                    logger.error(f"Error aggregating metric {metric_name} for {layer_identifier} in step(): {e}",
                                 exc_info=True)
                    if values_list: aggregated_value = values_list[0]

                if aggregated_value is not None:
                    current_step_processed_data[layer_identifier][metric_name] = aggregated_value

        if current_step_processed_data:
            self.processed_data_by_step[global_step] = current_step_processed_data
            logger.info(f"ActivityMonitor collected and processed data for step {global_step}.")

        self.hook_collected_buffer.clear()
        return wandb_metrics

    def get_data_for_step(self, global_step: int) -> Dict[str, Any]:
        return self.processed_data_by_step.get(global_step, {})

    def export_all_processed_data_to_records(self) -> List[Dict[str, Any]]:
        """
        Flattens all processed_data_by_step into a list of records for CSV export.
        """
        records = []
        for global_step, step_data in self.processed_data_by_step.items():
            for layer_identifier, metrics in step_data.items():
                for metric_name, value in metrics.items():
                    base_record = {
                        "global_step": global_step,
                        "layer_identifier": layer_identifier,
                        "original_metric_name": metric_name
                    }

                    np_value = None
                    if isinstance(value, torch.Tensor):
                        np_value = value.numpy()
                    elif isinstance(value, np.ndarray):
                        np_value = value
                    else:  # Scalar
                        records.append({**base_record, "metric_type": "scalar", "metric_value": float(value)})
                        continue

                    if np_value.ndim == 0:
                        records.append({**base_record, "metric_type": "scalar", "metric_value": float(np_value.item())})
                    elif metric_name == 'full_activation_map':
                        records.append(
                            {**base_record, "metric_type": "full_map_shape", "metric_value": str(np_value.shape)})
                        records.append({**base_record, "metric_type": "full_map_mean",
                                        "metric_value": float(np.mean(np_value.astype(np.float32)))})
                        records.append({**base_record, "metric_type": "full_map_std",
                                        "metric_value": float(np.std(np_value.astype(np.float32)))})
                        records.append({**base_record, "metric_type": "full_map_min",
                                        "metric_value": float(np.min(np_value.astype(np.float32)))})
                        records.append({**base_record, "metric_type": "full_map_max",
                                        "metric_value": float(np.max(np_value.astype(np.float32)))})
                    elif 'mean_abs_activation_per_channel' in metric_name:
                        records.append({**base_record, "metric_type": "per_channel_overall_mean",
                                        "metric_value": float(np.mean(np_value))})
                        records.append({**base_record, "metric_type": "per_channel_overall_std",
                                        "metric_value": float(np.std(np_value))})
                        records.append({**base_record, "metric_type": "per_channel_overall_min",
                                        "metric_value": float(np.min(np_value))})
                        records.append({**base_record, "metric_type": "per_channel_overall_max",
                                        "metric_value": float(np.max(np_value))})
                    else:  # Generic ndarray (e.g. mean_activation if it was an array, though unlikely with current _calculate_metrics)
                        records.append({**base_record, "metric_type": "array_mean",
                                        "metric_value": float(np.mean(np_value.astype(np.float32)))})
                        records.append({**base_record, "metric_type": "array_std",
                                        "metric_value": float(np.std(np_value.astype(np.float32)))})
        return records

    def __del__(self):
        self.remove_hooks()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    class DummyModelWithGroupNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.relu1 = torch.nn.ReLU()
            self.norm1 = torch.nn.GroupNorm(4, 8)
            self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
            self.relu2 = torch.nn.ReLU()
            self.norm2 = torch.nn.GroupNorm(8, 16)
            self.final_conv = torch.nn.Conv2d(16, 3, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.norm1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.norm2(x)
            x = self.final_conv(x)
            return x


    model = DummyModelWithGroupNorm()

    tracking_config_d1_2 = {
        "enabled": True,
        "target_layers": [
            {
                "name": "norm1",
                "capture_point": "input",
                "metrics": ["mean_abs_activation_per_channel", "full_activation_map", "mean_activation"]
            },
            {
                "name": "norm1",
                "capture_point": "output",
                "metrics": ["mean_abs_activation_per_channel", "std_activation"]
            }
        ],
        "track_interval": 1
    }

    logger.info("\n--- Initializing Monitor (CSV Export Test) ---")
    monitor = ActivityMonitor(model, tracking_config_d1_2)

    logger.info("\n--- Running Dummy Forward Pass (Batch 1, Step 1) ---")
    model(torch.randn(2, 3, 32, 32))

    logger.info("\n--- Calling Monitor Step (global_step=1) ---")
    wandb_logs_step_1 = monitor.step(global_step=1)
    print(f"WandB Logs at step 1: {wandb_logs_step_1}")
    assert "tracking/norm1.input/mean_abs_activation_per_channel_overall_mean" in wandb_logs_step_1
    assert "tracking/norm1.input/full_activation_map_mean" in wandb_logs_step_1, f"Missing full_activation_map_mean. Got: {wandb_logs_step_1.keys()}"
    assert "tracking/norm1.output/mean_abs_activation_per_channel_overall_mean" in wandb_logs_step_1

    logger.info("\n--- Running Dummy Forward Pass (Batch 1, Step 2) ---")
    model(torch.randn(2, 3, 32, 32) * 0.5)  # Different input

    logger.info("\n--- Calling Monitor Step (global_step=2) ---")
    wandb_logs_step_2 = monitor.step(global_step=2)
    print(f"WandB Logs at step 2: {wandb_logs_step_2}")

    logger.info("\n--- Exporting Data to Records ---")
    all_records = monitor.export_all_processed_data_to_records()
    print(f"Exported {len(all_records)} records.")
    if all_records:
        print("Sample records:")
        for i in range(min(5, len(all_records))):
            print(all_records[i])

    # Verify some expected records exist
    assert any(
        r['global_step'] == 1 and r['layer_identifier'] == 'norm1.input' and r['metric_type'] == 'full_map_shape' for r
        in all_records)
    assert any(r['global_step'] == 1 and r['layer_identifier'] == 'norm1.input' and r[
        'metric_type'] == 'per_channel_overall_mean' for r in all_records)
    assert any(r['global_step'] == 2 and r['layer_identifier'] == 'norm1.output' and r[
        'metric_type'] == 'per_channel_overall_mean' for r in all_records)

    logger.info("CSV export structure test completed.")
    del monitor
