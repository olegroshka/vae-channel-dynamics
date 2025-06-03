# src/tracking/deadneuron.py
import logging
import torch
import torch.nn as nn
from typing import List, Tuple, Type
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeadNeuronTracker:

    def __init__(self,
                 target_layer_classes: Tuple[Type[nn.Module], ...],
                 target_layer_names_for_raw_weights: List[str],
                 threshold: float,
                 mean_percentage: float,
                 dead_type: str = "threshold"):
        self.threshold = threshold
        self.mean_percentage = mean_percentage
        self.target_layer_classes = target_layer_classes
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

        self.weights_history = defaultdict(list)
        self.percent_history = defaultdict(list)

    def track_dead_neurons(self, model_wrapper: nn.Module, global_step: int):
        actual_vae_model = None
        if hasattr(model_wrapper, 'vae') and model_wrapper.vae is not None:
            actual_vae_model = model_wrapper.vae
        elif isinstance(model_wrapper, nn.Module):
            actual_vae_model = model_wrapper
        else:
            logger.error("DeadNeuronTracker: model_wrapper is not an nn.Module or has no .vae attribute.")
            return
        if actual_vae_model is None:
            logger.error("DeadNeuronTracker: actual_vae_model is None.")
            return

        for name, param in actual_vae_model.named_parameters():
            if not param.requires_grad:
                continue

            if name in self.target_layer_names_for_raw_weights:
                self.weights_history[name] = [param.detach().cpu().numpy()]

            if "weight" in name or "bias" in name:
                module_path = ".".join(name.split(".")[:-1])
                try:
                    module = actual_vae_model.get_submodule(module_path)
                    if isinstance(module, self.target_layer_classes):
                        percentage = self.get_percentage(param)
                        self.percent_history[name].append((global_step, percentage))
                        # Log only if there's a notable percentage or for specific debug
                        if percentage > 0.1 or logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"DeadNeuronTracker (step {global_step}): Tracked {name}, Type: {type(module).__name__}, Dead %: {percentage:.2f}%")
                except AttributeError:
                    continue
                except Exception as e:
                    logger.error(
                        f"DeadNeuronTracker (step {global_step}): Error processing param {name} for percentage: {e}")

    def noop(self, param):
        return 0.0

    def smaller_than_threshold(self, param: torch.Tensor) -> float:
        total_elements = param.numel()
        if total_elements == 0: return 0.0
        small_values = (param.abs() < self.threshold).sum().item()
        return (small_values / total_elements) * 100.0

    def percent_of_mean(self, param: torch.Tensor) -> float:
        if param.numel() == 0: return 0.0
        param_abs = param.abs()
        mean_abs = param_abs.mean().item()
        if abs(mean_abs) < 1e-9:
            is_all_zero = (param_abs < 1e-9).all().item()
            return 100.0 if is_all_zero else 0.0
        adaptive_threshold = self.mean_percentage * mean_abs
        small_values = (param_abs < adaptive_threshold).sum().item()
        total_elements = param.numel()
        return (small_values / total_elements) * 100.0

    def both(self, param: torch.Tensor) -> float:
        total_elements = param.numel()
        if total_elements == 0: return 0.0

        param_abs = param.abs()
        condition_fixed = param_abs < self.threshold

        mean_abs = param_abs.mean().item()

        if abs(mean_abs) < 1e-9:
            # If mean is effectively zero, adaptive condition is essentially checking if param is also zero.
            # A very small non-zero param would not be < (0.1 * effectively_zero_mean_if_not_truly_zero)
            condition_adaptive = param_abs < 1e-9  # Check if param itself is effectively zero
        else:
            adaptive_threshold = self.mean_percentage * mean_abs
            condition_adaptive = param_abs < adaptive_threshold

        combined_condition = condition_fixed & condition_adaptive
        small_values = combined_condition.sum().item()
        return (small_values / total_elements) * 100.0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


    class DummyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # 3*8*3*3 = 216 elements
            self.conv1.weight.data.fill_(0.001)
            self.conv1.weight.data[0, 0, 0, 0] = 1.0
            self.conv1.weight.data[1, 0, 0, 0] = 1e-7  # Strictly less than 1e-5

            self.gn1 = nn.GroupNorm(2, 8)  # 8 weights, 8 biases
            # For step 0:
            self.gn1.weight.data.fill_(1e-6)  # Strictly less than 1e-5 (threshold)
            # mean_abs = 1e-6. adaptive_thresh = 0.1 * 1e-6 = 1e-7.
            # 1e-6 < 1e-7 is False. So "both" should be 0%.
            self.gn1.bias.data.fill_(1e-7)  # Strictly less than 1e-5.
            # mean_abs = 1e-7. adaptive_thresh = 0.1 * 1e-7 = 1e-8.
            # 1e-7 < 1e-8 is False. So "both" should be 0%.

            self.fc1 = nn.Linear(10, 2)
            self.fc1.weight.data.normal_(0, 0.01)
            self.fc1.bias.data.normal_(0, 0.01)

            self.another_conv = nn.Conv2d(8, 4, kernel_size=1)
            self.another_conv.weight.data.fill_(0.5)
            self.another_conv.bias.data.fill_(0.1)


    class ModelWrapper(nn.Module):
        def __init__(self): super().__init__(); self.vae = DummyVAE()


    test_model_wrapper = ModelWrapper()
    TARGET_CLASSES = (nn.Conv2d, nn.Linear, nn.GroupNorm)
    TARGET_NAMES_RAW = ["another_conv.weight", "gn1.weight", "gn1.bias"]
    THRESHOLD_VAL = 1e-5  # This is self.threshold
    MEAN_PERCENT_VAL = 0.1  # This is self.mean_percentage

    tracker = DeadNeuronTracker(TARGET_CLASSES, TARGET_NAMES_RAW, THRESHOLD_VAL, MEAN_PERCENT_VAL, "both")

    print("--- Tracking at step 0 ---")
    tracker.track_dead_neurons(test_model_wrapper, global_step=0)

    print("--- Tracking at step 20 ---")
    # Change weights to be clearly NOT dead
    test_model_wrapper.vae.conv1.weight.data.fill_(1.0)
    test_model_wrapper.vae.gn1.weight.data.fill_(1.0)
    test_model_wrapper.vae.gn1.bias.data.fill_(0.5)
    test_model_wrapper.vae.fc1.weight.data.fill_(1.0)
    tracker.track_dead_neurons(test_model_wrapper, global_step=20)

    print("\nPercent History (dead_type='both'):")
    for layer, history in tracker.percent_history.items():
        print(f"  {layer}: {history}")

    print("\nWeights History (should have latest snapshot for targeted raw weights):")
    for layer, history_list in tracker.weights_history.items():
        print(f"  {layer}: shape {history_list[0].shape if history_list else 'N/A'}")

    # Assertions based on the "both" logic and initial values:
    # conv1.weight at step 0: one element is 1e-7.
    #   Fixed: 1e-7 < 1e-5 (True). Mean is approx (1e-7 + 215*0.001)/216 = 0.00099. Adaptive = 0.1 * 0.00099 = 9.9e-5.
    #   Adaptive: 1e-7 < 9.9e-5 (True). So 1 element is dead. (1/216*100 = 0.46%)
    assert 'conv1.weight' in tracker.percent_history
    conv1_w_hist = tracker.percent_history['conv1.weight']
    assert conv1_w_hist[0] == (0, (1 / 216) * 100.0)
    assert conv1_w_hist[1] == (20, 0.0)  # All 1.0 at step 20

    # gn1.weight at step 0: all elements 1e-6.
    #   Fixed: 1e-6 < 1e-5 (True). Mean is 1e-6. Adaptive = 0.1 * 1e-6 = 1e-7.
    #   Adaptive: 1e-6 < 1e-7 (False). So 0 elements dead.
    assert 'gn1.weight' in tracker.percent_history
    gn1_w_hist = tracker.percent_history['gn1.weight']
    assert gn1_w_hist[0] == (0, 0.0)  # Corrected: Expected 0% dead
    assert gn1_w_hist[1] == (20, 0.0)  # Corrected: Expected 0% dead

    # gn1.bias at step 0: all elements 1e-7.
    #   Fixed: 1e-7 < 1e-5 (True). Mean is 1e-7. Adaptive = 0.1 * 1e-7 = 1e-8.
    #   Adaptive: 1e-7 < 1e-8 (False). So 0 elements dead.
    assert 'gn1.bias' in tracker.percent_history
    gn1_b_hist = tracker.percent_history['gn1.bias']
    assert gn1_b_hist[0] == (0, 0.0)  # Corrected: Expected 0% dead
    assert gn1_b_hist[1] == (20, 0.0)  # Corrected: Expected 0% dead (values are 0.5 at step 20)

    print("All assertions passed.")
