# src/tracking/deadneuron.py
import logging

import torch
import torch.nn as nn
from typing import List, Type, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeadNeuronTracker:

    def __init__(self,
                 target_layer_classes: Tuple[Type[nn.Module], ...],
                 target_layer_names: List[str],
                 threshold: float,
                 mean_percentage: float,
                 dead_type: str = "threshold"):
        """Tracks percentage of 'dead' neurons in specified layers of a model."""
        self.threshold = threshold
        self.mean_percentage = mean_percentage
        self.target_layer_classes = target_layer_classes
        self.target_layer_names = target_layer_names

        if dead_type == "threshold":
            self.get_percentage = self.smaller_than_threshold
        elif dead_type == "percent_of_mean":
            self.get_percentage = self.percent_of_mean
        elif dead_type == "both":
            self.get_percentage = self.both
        else:
            self.get_percentage = self.noop

        self.weights_history = defaultdict(list)
        self.percent_history = defaultdict(list)


    def track_dead_neurons(self, model: nn.Module):
        """
        Tracks the percentage of near-zero weights in specified layers of the model.
        A layer is considered to have 'dead' neurons if weights are below a threshold.
        """
        vae = model.vae
        for name, param in vae.named_parameters():

            if name in self.target_layer_names:
                self.weights_history[name].append(param.detach().cpu().numpy())

            if "weight" in name and param.requires_grad:
                module_path = ".".join(name.split(".")[:-1])
                try:
                    module = vae.get_submodule(module_path)

                    if isinstance(module, self.target_layer_classes):
                        self.percent_history[name].append(self.get_percentage(param))

                except AttributeError:
                    logger.debug(f"Could not get submodule for {name}, skipping dead neuron check.")
                    continue
                except Exception as e:
                    logger.error(f"Error checking dead neurons for {name}: {e}")

    def noop(self, param):
        return 0

    def smaller_than_threshold(self, param):
        total_elements = param.numel()
        if total_elements == 0:
            return 0
        small_values = (param.abs() < self.threshold).sum().item()
        return (small_values / total_elements) * 100

    def percent_of_mean(self, param):
        param_abs = param.abs()
        mean_abs = param_abs.mean().item()
        if mean_abs == 0:
            return 0
        adaptive_threshold = self.mean_percentage * mean_abs
        small_values = (param_abs < adaptive_threshold).sum().item()
        total_elements = param.numel()
        return (small_values / total_elements) * 100

    def both(self, param):
        total_elements = param.numel()
        if total_elements == 0:
            return 0

        param_abs = param.abs()
        mean_abs = param_abs.mean().item()

        condition_fixed = param_abs < self.threshold

        if mean_abs == 0:
            condition_adaptive = torch.zeros_like(condition_fixed, dtype=torch.bool)  # no elements satisfy adaptive if mean=0
        else:
            adaptive_threshold = self.mean_percentage * mean_abs
            condition_adaptive = param_abs < adaptive_threshold

        combined_condition = condition_fixed & condition_adaptive

        small_values = combined_condition.sum().item()
        return (small_values / total_elements) * 100
