# src/analysis/logit_lens.py
import os
import logging
from typing import Dict, Any, List, Optional, Callable

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
            nn.ConvTranspose2d(self.config["mini_decoder_input_channels"], 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Example: upscale
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Example: to RGB
            nn.Sigmoid()  # Output to [0,1] for image
        )
        logger.info("Placeholder mini-decoder initialized.")

    def _get_safe_layer_name(self, layer_identifier: str) -> str:
        """Converts a layer identifier string into a filesystem-safe name."""
        return layer_identifier.replace(".", "_").replace("/", "_")

    def get_layer_logit_length(self, activation_map_tensor: torch.Tensor, layer_identifier: str) -> Optional[int]:
        """
        Computes and logs the "logit length" (number of channels) of a given layer's
        activation map. For Logit Lens, this is the number of channels being processed.

        Args:
            activation_map_tensor: A 4D PyTorch tensor (Batch, Channels, Height, Width)
                                   containing the activation maps.
            layer_identifier: String identifier for the layer.

        Returns:
            The number of channels (logit length) if the tensor is 4D, otherwise None.
        """
        if not isinstance(activation_map_tensor, torch.Tensor) or activation_map_tensor.ndim != 4:
            logger.warning(
                f"Cannot compute logit length for {layer_identifier}: activation map is not a 4D tensor. Shape: {activation_map_tensor.shape if hasattr(activation_map_tensor, 'shape') else 'N/A'}")
            return None

        num_channels = activation_map_tensor.shape[1]
        logger.info(f"Logit length (number of channels) for layer '{layer_identifier}': {num_channels}")
        return num_channels


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

        self.get_layer_logit_length(activation_map_tensor, layer_identifier) # Log the logit length

        samples_to_process = min(_num_samples, batch_size)
        channels_to_process = min(_num_channels, total_channels)

        safe_layer_name = self._get_safe_layer_name(layer_identifier)
        output_subdir = os.path.join(self.visualization_base_dir, f"step_{global_step}", safe_layer_name)
        os.makedirs(output_subdir, exist_ok=True)

        for sample_idx in range(samples_to_process):
            fig, axes = plt.subplots(1, channels_to_process, figsize=(channels_to_process * 4, 4))
            if channels_to_process == 1: # Handle the case of a single subplot
                axes = [axes]

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
                        normalized_map = torch.zeros_like(channel_map)

                    # Plot on the subplot
                    axes[channel_idx].imshow(normalized_map.numpy(), cmap=colormap)
                    axes[channel_idx].set_title(f'Channel {channel_idx}')
                    axes[channel_idx].axis('off') # Hide axes ticks

                except Exception as e:
                    logger.error(
                        f"Error visualizing map for {layer_identifier}, sample {sample_idx}, channel {channel_idx}: {e}",
                        exc_info=True)

            plt.tight_layout()
            save_path = os.path.join(output_subdir, f"sample_{sample_idx}_all_channels.png")
            plt.savefig(save_path)
            plt.close(fig)
            logger.info(f"Saved combined activation map visualization for {layer_identifier}, sample {sample_idx} to {save_path}")


    def run_logit_lens_with_activations(self,
                                        global_step: int,
                                        layers_to_analyze: List[str],
                                        num_batch_samples_to_viz: Optional[int],
                                        projection_type: str,
                                        activations_to_process: Dict[str, torch.Tensor]):
        """
        Performs a "Logit Lens" analysis by taking *provided* intermediate activations
        and projecting them through a specified "lens" (e.g., mini-decoder).

        Args:
            global_step: Current training step for output file naming.
            layers_to_analyze: List of layer names whose activations should be in `activations_to_process`.
            num_batch_samples_to_viz: Number of batch samples to project.
            projection_type: Defines how the projection is done.
                             - "mini_decoder_single_channel": Projects one channel at a time via mini_decoder.
                             - "mini_decoder_full_map": Projects the full activation map through the mini_decoder.
            activations_to_process: A dictionary where keys are layer names (str) and values are
                                    the captured activation tensors (B, C, H, W).
        """
        _num_samples = num_batch_samples_to_viz if num_batch_samples_to_viz is not None else self.default_batch_samples

        logger.info(f"\n--- Running Logit Lens for step {global_step} ---")

        if not activations_to_process:
            logger.warning("No activations provided to run_logit_lens_with_activations. Skipping.")
            return

        for layer_name in layers_to_analyze:
            if layer_name not in activations_to_process:
                logger.warning(f"No activation found for layer '{layer_name}' in provided dict. Skipping.")
                continue

            activation_map = activations_to_process[layer_name] # (B, C, H, W)
            batch_size, total_channels, height, width = activation_map.shape
            samples_to_process = min(_num_samples, batch_size)

            safe_layer_name = self._get_safe_layer_name(layer_name)
            output_subdir = os.path.join(self.visualization_base_dir, f"step_{global_step}", safe_layer_name, "logit_lens_projections")
            os.makedirs(output_subdir, exist_ok=True)

            logger.info(f"Processing Logit Lens for layer '{layer_name}' with shape {activation_map.shape}")

            for sample_idx in range(samples_to_process):
                try:
                    if projection_type == "mini_decoder_single_channel":
                        channels_to_project = min(self.default_num_channels, total_channels)
                        # Instead of saving individual images, create a single figure for all projected channels
                        fig_proj, axes_proj = plt.subplots(1, channels_to_project, figsize=(channels_to_project * 4, 4))
                        if channels_to_project == 1:
                            axes_proj = [axes_proj] # Ensure axes is iterable for single subplot case

                        for channel_idx in range(channels_to_project):
                            single_channel_map = activation_map[sample_idx, channel_idx, :, :].unsqueeze(0).unsqueeze(0)
                            projected_img = self._project_through_mini_decoder(single_channel_map)

                            # Convert to NumPy array and transpose for matplotlib (H, W, C)
                            img_np = projected_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            axes_proj[channel_idx].imshow(img_np)
                            axes_proj[channel_idx].set_title(f'Proj. Ch. {channel_idx}')
                            axes_proj[channel_idx].axis('off')

                        plt.tight_layout()
                        save_path = os.path.join(output_subdir, f"lens_sample_{sample_idx}_single_channel_projections_combined.png")
                        plt.savefig(save_path)
                        plt.close(fig_proj)
                        logger.debug(f"Saved combined single-channel projections for {layer_name}, sample {sample_idx}")

                    elif projection_type == "mini_decoder_full_map":
                        full_map_tensor = activation_map[sample_idx:sample_idx+1, :, :, :]
                        if full_map_tensor.shape[1] != self.mini_decoder[0].in_channels:
                            logger.warning(f"Mismatch: Mini-decoder expects {self.mini_decoder[0].in_channels} input channels, "
                                           f"but layer '{layer_name}' has {full_map_tensor.shape[1]} channels. Skipping full map projection.")
                            continue
                        projected_img = self._project_through_mini_decoder(full_map_tensor)

                        # Convert to NumPy array and transpose for matplotlib (H, W, C)
                        img_np = projected_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        fig_full, ax_full = plt.subplots(1, 1, figsize=(6, 6))
                        ax_full.imshow(img_np)
                        ax_full.set_title(f'Full Map Projection for Sample {sample_idx}')
                        ax_full.axis('off')
                        plt.tight_layout()
                        save_path = os.path.join(output_subdir, f"lens_sample_{sample_idx}_full_map_projection.png")
                        plt.savefig(save_path)
                        plt.close(fig_full)
                        logger.debug(f"Saved full map projection for {layer_name}, sample {sample_idx}")
                    else:
                        logger.warning(f"Unknown projection_type: {projection_type}. Skipping.")

                except Exception as e:
                    logger.error(f"Error during Logit Lens projection for layer '{layer_name}', sample {sample_idx}: {e}", exc_info=True)
        logger.info(f"Logit Lens analysis completed for step {global_step}.")


    def _project_through_mini_decoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to run a tensor through the mini_decoder."""
        # Ensure mini_decoder is on the same device as the input tensor
        if self.mini_decoder[0].weight.device != input_tensor.device:
            input_tensor = input_tensor.to(self.mini_decoder[0].weight.device)
        with torch.no_grad():
            projected_patch = self.mini_decoder(input_tensor)
        return projected_patch


    def project_with_mini_decoder(self, # Keeping this for backwards compatibility/specific single-channel use
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

    def run_logit_lens_with_activations(self,
                                        global_step: int,
                                        layers_to_analyze: List[str],
                                        num_batch_samples_to_viz: Optional[int],
                                        projection_type: str,
                                        activations_to_process: Dict[str, torch.Tensor]):
        """
        Performs a "Logit Lens" analysis by taking *provided* intermediate activations
        and projecting them through a specified "lens" (e.g., mini-decoder).

        Args:
            global_step: Current training step for output file naming.
            layers_to_analyze: List of layer names whose activations should be in `activations_to_process`.
            num_batch_samples_to_viz: Number of batch samples to project.
            projection_type: Defines how the projection is done.
                             - "mini_decoder_single_channel": Projects one channel at a time via mini_decoder.
                             - "mini_decoder_full_map": Projects the full activation map through the mini_decoder.
            activations_to_process: A dictionary where keys are layer names (str) and values are
                                    the captured activation tensors (B, C, H, W) (expected on CPU).
        """
        _num_samples = num_batch_samples_to_viz if num_batch_samples_to_viz is not None else self.default_batch_samples

        logger.info(f"\n--- Running Logit Lens for step {global_step} ---")

        if not activations_to_process:
            logger.warning("No activations provided to run_logit_lens_with_activations. Skipping.")
            return

        for layer_name in layers_to_analyze: # Loop through the layers you want to analyze
            if layer_name not in activations_to_process:
                logger.warning(f"No activation found for layer '{layer_name}' in provided dict. Skipping.")
                continue

            activation_map = activations_to_process[layer_name] # Get the activation tensor for this layer
            batch_size, total_channels, height, width = activation_map.shape
            samples_to_process = min(_num_samples, batch_size)

            safe_layer_name = self._get_safe_layer_name(layer_name)
            # Create output directory structure
            output_subdir = os.path.join(self.visualization_base_dir, f"step_{global_step}", safe_layer_name, "logit_lens_projections")
            os.makedirs(output_subdir, exist_ok=True)

            logger.info(f"Processing Logit Lens for layer '{layer_name}' with shape {activation_map.shape}")

            for sample_idx in range(samples_to_process): # Loop through batch samples
                try:
                    if projection_type == "mini_decoder_single_channel":
                        channels_to_project = min(self.default_num_channels, total_channels)
                        # This creates ONE figure for all projected channels of the current sample_idx
                        fig_proj, axes_proj = plt.subplots(1, channels_to_project, figsize=(channels_to_project * 4, 4))
                        if channels_to_project == 1:
                            axes_proj = [axes_proj]

                        for channel_idx in range(channels_to_project):
                            # Select a single channel and add batch and channel dimension: (1, 1, H, W)
                            single_channel_map = activation_map[sample_idx, channel_idx, :, :].unsqueeze(0).unsqueeze(0)
                            # Pass this single channel through the mini_decoder
                            img_np = self._project_through_mini_decoder(single_channel_map).squeeze(0).permute(1, 2, 0)
                            # Plotting on the subplot of the ONE figure
                            axes_proj[channel_idx].imshow(img_np)
                            axes_proj[channel_idx].set_title(f'Proj. Ch. {channel_idx}')
                            axes_proj[channel_idx].axis('off')

                        plt.tight_layout()
                        # This saves the ONE figure
                        save_path = os.path.join(output_subdir, f"lens_sample_{sample_idx}_single_channel_projections_combined.png")
                        plt.savefig(save_path)
                        plt.close(fig_proj)
                        logger.debug(f"Saved combined single-channel projections for {layer_name}, sample {sample_idx}")

                    elif projection_type == "mini_decoder_full_map":
                        # Select the full activation map for one sample: (1, C, H, W)
                        full_map_tensor = activation_map[sample_idx:sample_idx+1, :, :, :]
                        # Important check: does the mini_decoder have the right input channels?
                        if full_map_tensor.shape[1] != self.mini_decoder[0].in_channels:
                            logger.warning(f"Mismatch: Mini-decoder expects {self.mini_decoder[0].in_channels} input channels, "
                                           f"but layer '{layer_name}' has {full_map_tensor.shape[1]} channels. Skipping full map projection.")
                            continue
                        # Pass the entire activation map through the mini_decoder
                        projected_img = self._project_through_mini_decoder(full_map_tensor)
                        # Save the resulting image
                        save_path = os.path.join(output_subdir, f"lens_sample_{sample_idx}_full_map.png")
                        TF.to_pil_image(projected_img.squeeze(0)).save(save_path)
                        logger.debug(f"Saved full map projection for {layer_name}, sample {sample_idx}")
                    else:
                        logger.warning(f"Unknown projection_type: {projection_type}. Skipping.")

                except Exception as e:
                    logger.error(f"Error during Logit Lens projection for layer '{layer_name}', sample {sample_idx}: {e}", exc_info=True)
        logger.info(f"Logit Lens analysis completed for step {global_step}.")

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

