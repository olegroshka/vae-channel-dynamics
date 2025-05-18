# src/analysis/logit_lens.py
import os
import logging
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class VAELogitLens:
    """
    A class to analyze and visualize activation maps from VAE layers,
    particularly pre-normalization activations, to understand what features
    channels might be representing.
    """

    def __init__(self,
                 model_for_lens: Optional[nn.Module] = None,  # VAE model or parts of it
                 logit_lens_config: Optional[Dict[str, Any]] = None,
                 main_experiment_output_dir: str = "./experiment_outputs"):
        """
        Initializes the VAELogitLens.

        Args:
            model_for_lens: The VAE model (or relevant parts) if needed for
                            more advanced projections. Not used in basic visualization.
            logit_lens_config: Configuration dictionary for the logit lens.
                               Expected keys:
                               - visualization_output_subdir (str): Subdirectory for saving visualizations.
                               - default_num_channels_to_viz (int): Default number of channels to show.
                               - default_num_batch_samples_to_viz (int): Default batch samples to show.
            main_experiment_output_dir: The main output directory for the experiment.
        """
        self.model = model_for_lens  # Store if needed later
        self.config = logit_lens_config if logit_lens_config is not None else {}

        self.default_num_channels = self.config.get("default_num_channels_to_viz", 4)
        self.default_batch_samples = self.config.get("default_num_batch_samples_to_viz", 1)

        viz_subdir = self.config.get("visualization_output_subdir", "logit_lens_visualizations")
        self.visualization_base_dir = os.path.join(main_experiment_output_dir, viz_subdir)
        os.makedirs(self.visualization_base_dir, exist_ok=True)

        logger.info(f"VAELogitLens initialized. Visualizations will be saved in: {self.visualization_base_dir}")

        # Placeholder for a mini-decoder (inspired by LLM Logit Lens output projection)
        # This is a very basic example, input channels would depend on the activation map.
        # For a single channel input (H, W), it would need to be unsqueezed.
        self.mini_decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Example: upscale
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Example: to RGB
            nn.Sigmoid()  # Output to [0,1] for image
        )
        logger.info("Placeholder mini-decoder initialized.")

    def _get_safe_layer_name(self, layer_identifier: str) -> str:
        """Converts a layer identifier string into a filesystem-safe name."""
        return layer_identifier.replace(".", "_").replace("/", "_")

    def visualize_channel_activation_maps(
            self,
            activation_map_tensor: torch.Tensor,
            layer_identifier: str,
            global_step: int,
            num_channels_to_viz: Optional[int] = None,
            num_batch_samples_to_viz: Optional[int] = None,
            colormap: str = 'viridis'
    ):
        """
        Visualizes and saves spatial activation maps for selected channels.

        Args:
            activation_map_tensor: A 4D PyTorch tensor (Batch, Channels, Height, Width)
                                   containing the activation maps (expected on CPU).
            layer_identifier: String identifier for the layer (e.g., "vae.encoder.norm1.input").
            global_step: The current global training step, for naming output files.
            num_channels_to_viz: Number of channels to visualize from the map.
                                 Defaults to self.default_num_channels.
            num_batch_samples_to_viz: Number of batch samples to visualize.
                                      Defaults to self.default_batch_samples.
            colormap: Matplotlib colormap to use for visualization.
        """
        if not isinstance(activation_map_tensor, torch.Tensor) or activation_map_tensor.ndim != 4:
            logger.warning(
                f"Activation map for {layer_identifier} is not a 4D tensor. Shape: {activation_map_tensor.shape if hasattr(activation_map_tensor, 'shape') else 'N/A'}. Skipping visualization.")
            return

        _num_channels = num_channels_to_viz if num_channels_to_viz is not None else self.default_num_channels
        _num_samples = num_batch_samples_to_viz if num_batch_samples_to_viz is not None else self.default_batch_samples

        batch_size, total_channels, height, width = activation_map_tensor.shape

        samples_to_process = min(_num_samples, batch_size)
        channels_to_process = min(_num_channels, total_channels)

        safe_layer_name = self._get_safe_layer_name(layer_identifier)
        output_subdir = os.path.join(self.visualization_base_dir, f"step_{global_step}", safe_layer_name)
        os.makedirs(output_subdir, exist_ok=True)

        for sample_idx in range(samples_to_process):
            for channel_idx in range(channels_to_process):
                try:
                    # Extract single channel map (H, W)
                    channel_map = activation_map_tensor[sample_idx, channel_idx, :, :]

                    # Normalize the map to [0, 1] for visualization
                    map_min = channel_map.min()
                    map_max = channel_map.max()
                    if map_max - map_min > 1e-6:  # Avoid division by zero if map is flat
                        normalized_map = (channel_map - map_min) / (map_max - map_min)
                    else:
                        normalized_map = torch.zeros_like(channel_map)  # or ones_like, or assign map_min

                    # Convert to PIL Image using a colormap
                    # plt.imsave expects a NumPy array
                    pil_img = Image.fromarray(
                        (plt.cm.get_cmap(colormap)(normalized_map.numpy())[:, :, :3] * 255).astype(np.uint8))

                    # Alternative: Convert directly to grayscale PIL Image
                    # pil_img = TF.to_pil_image(normalized_map.unsqueeze(0)) # Add channel dim for grayscale

                    save_path = os.path.join(output_subdir, f"sample_{sample_idx}_channel_{channel_idx}.png")
                    pil_img.save(save_path)
                except Exception as e:
                    logger.error(
                        f"Error visualizing map for {layer_identifier}, sample {sample_idx}, channel {channel_idx}: {e}",
                        exc_info=True)

        logger.info(
            f"Saved {samples_to_process * channels_to_process} activation map visualizations for {layer_identifier} at step {global_step} to {output_subdir}")

    def project_with_mini_decoder(self,
                                  activation_map_tensor: torch.Tensor,
                                  layer_identifier: str,
                                  global_step: int,
                                  channel_idx: int = 0,
                                  sample_idx: int = 0):
        """
        Placeholder: Takes a single channel's activation map, passes it through
        a mini-decoder, and saves the resulting "image patch".

        Args:
            activation_map_tensor: 4D activation tensor (B, C, H, W).
            layer_identifier: Identifier for the source layer.
            global_step: Current training step.
            channel_idx: Index of the channel to project.
            sample_idx: Index of the batch sample to use.
        """
        if not (0 <= sample_idx < activation_map_tensor.shape[0] and 0 <= channel_idx < activation_map_tensor.shape[1]):
            logger.warning(f"Invalid sample_idx or channel_idx for mini-decoder projection. Skipping.")
            return

        single_channel_map = activation_map_tensor[sample_idx, channel_idx, :, :].unsqueeze(0).unsqueeze(
            0)  # Shape (1, 1, H, W)

        # Ensure mini_decoder is on the same device as the input tensor (or move input to CPU if decoder is on CPU)
        # Assuming mini_decoder is on CPU for now.
        single_channel_map = single_channel_map.cpu()
        self.mini_decoder.cpu()  # Ensure mini_decoder is on CPU

        try:
            with torch.no_grad():
                projected_patch = self.mini_decoder(single_channel_map)  # Output shape (1, 3, H', W')

            # Save the projected patch
            safe_layer_name = self._get_safe_layer_name(layer_identifier)
            output_subdir = os.path.join(self.visualization_base_dir, f"step_{global_step}", safe_layer_name,
                                         "mini_decoded")
            os.makedirs(output_subdir, exist_ok=True)

            # Convert to PIL image (assuming output is [0,1] due to Sigmoid)
            img_tensor_to_save = projected_patch.squeeze(0)  # Remove batch dim -> (3, H', W')
            pil_img = TF.to_pil_image(img_tensor_to_save)

            save_path = os.path.join(output_subdir, f"sample_{sample_idx}_channel_{channel_idx}_projected.png")
            pil_img.save(save_path)
            logger.info(
                f"Saved mini-decoder projection for {layer_identifier}, sample {sample_idx}, channel {channel_idx} to {save_path}")

        except Exception as e:
            logger.error(f"Error during mini-decoder projection for {layer_identifier}: {e}", exc_info=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- Configuration for testing ---
    test_output_dir = "./test_experiment_outputs"
    lens_config = {
        "visualization_output_subdir": "logit_lens_visualizations_test",
        "default_num_channels_to_viz": 2,
        "default_num_batch_samples_to_viz": 1
    }

    # --- Instantiate VAELogitLens ---
    logit_lens_analyzer = VAELogitLens(logit_lens_config=lens_config, main_experiment_output_dir=test_output_dir)

    # --- Test visualize_channel_activation_maps ---
    logger.info("\n--- Testing visualize_channel_activation_maps ---")
    # Dummy activation map (Batch=1, Channels=4, Height=16, Width=16)
    dummy_activations = torch.rand(1, 4, 16, 16) * 10 - 5  # Random values
    dummy_activations[0, 0, :8, :8] = 20  # Make one channel/region distinct
    dummy_activations[0, 1, 8:, 8:] = -10  # Make another distinct

    logit_lens_analyzer.visualize_channel_activation_maps(
        activation_map_tensor=dummy_activations,
        layer_identifier="dummy_layer.input",
        global_step=0,
        num_channels_to_viz=2,  # Test overriding default
        num_batch_samples_to_viz=1
    )

    # Test with more channels/samples than available
    logit_lens_analyzer.visualize_channel_activation_maps(
        activation_map_tensor=dummy_activations,
        layer_identifier="dummy_layer.another_input",
        global_step=0,
        num_channels_to_viz=8,  # More than available
        num_batch_samples_to_viz=2  # More than available
    )

    # --- Test project_with_mini_decoder (placeholder) ---
    logger.info("\n--- Testing project_with_mini_decoder ---")
    # Adjust mini_decoder input channels if needed, or ensure dummy_activations has appropriate channels.
    # The current mini_decoder expects 1 input channel.
    # Let's resize dummy_activations for mini_decoder input (e.g. if it's 1xHxW)
    # For this test, let's assume the mini_decoder is for a 16x16 input that gets upscaled.
    # The mini_decoder is defined with ConvTranspose2d(1, ...), so it expects 1 input channel.

    # We need to adjust the mini_decoder's first layer's input channels based on what it receives.
    # Or, for the test, ensure we pass a single channel.
    # For the test, we'll pass a single channel from dummy_activations.
    # The `project_with_mini_decoder` already selects a single channel.

    # Modify mini_decoder input channels for testing if needed, or ensure input matches.
    # For this example, let's assume the mini_decoder is designed for a HxW input (1 channel).
    # The provided dummy_activations are (B, C, H, W).
    # The project_with_mini_decoder method will take one channel.

    # To make the mini_decoder work with a 16x16 input channel:
    # If the first ConvTranspose2d in mini_decoder is, for example, ConvTranspose2d(1, 16, ...),
    # and the input to project_with_mini_decoder is a single channel from a 16x16 feature map,
    # it should work. The current mini_decoder is designed for this.

    logit_lens_analyzer.project_with_mini_decoder(
        activation_map_tensor=dummy_activations,  # Pass the full 4D tensor
        layer_identifier="dummy_layer.input",
        global_step=0,
        channel_idx=0,  # Project the first channel
        sample_idx=0  # From the first batch sample
    )

    logger.info(
        f"\nCheck the '{test_output_dir}/{lens_config['visualization_output_subdir']}' directory for saved images.")

