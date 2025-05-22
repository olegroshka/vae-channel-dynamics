# src/tracking/deadneuron.py
import logging

import torch.nn as nn
from typing import List, Type, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DeadNeuronTracker:

    def __init__(self, threshold: float, target_layer_classes: Tuple[Type[nn.Module], ...], target_layer_names: List[str]):
        """Tracks percentage of 'dead' neurons in specified layers of a model."""
        self.threshold = threshold
        self.target_layer_classes = target_layer_classes
        self.target_layer_names = target_layer_names

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
                        total_elements = param.numel()
                        if total_elements == 0:
                            self.percent_history[name].append(0)
                        else:
                            small_values = (param.abs() < self.threshold).sum().item()
                            percentage = (small_values / total_elements) * 100
                            self.percent_history[name].append(percentage)

                except AttributeError:
                    logger.debug(f"Could not get submodule for {name}, skipping dead neuron check.")
                    continue
                except Exception as e:
                    logger.error(f"Error checking dead neurons for {name}: {e}")
