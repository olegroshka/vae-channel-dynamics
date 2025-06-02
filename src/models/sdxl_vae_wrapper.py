# src/models/sdxl_vae_wrapper.py
import logging

import torch
from diffusers import AutoencoderKL
from typing import Optional, Union, Dict, List, Callable

logger = logging.getLogger(__name__)

class SDXLVAEWrapper(torch.nn.Module):
    """
    A wrapper around the Hugging Face AutoencoderKL model (specifically for SDXL VAE)
    to facilitate loading, potential modifications, and integration into the training loop.
    """
    def __init__(self,
                 pretrained_model_name_or_path: str = "stabilityai/sdxl-vae",
                 torch_dtype: Optional[Union[str, torch.dtype]] = None):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.torch_dtype = torch_dtype
        self.vae = self._load_vae()
        self.scaling_factor = self.vae.config.scaling_factor

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_activations: Dict[str, torch.Tensor] = {} # Stores activations by layer name

    def _load_vae(self) -> AutoencoderKL:
        """Loads the AutoencoderKL model from Hugging Face Hub or local path."""
        logger.info(f"Loading VAE model from: {self.pretrained_model_name_or_path}")
        try:
            vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_name_or_path,
                torch_dtype=self.torch_dtype
            )
            logger.info("VAE model loaded successfully.")
            logger.info(f"VAE scaling factor: {vae.config.scaling_factor}")
            return vae
        except Exception as e:
            logger.error(f"Failed to load VAE model from {self.pretrained_model_name_or_path}: {e}")
            raise

    def forward(self, pixel_values: torch.Tensor, sample_posterior: bool = True):
        """
        Defines the forward pass for VAE training or inference.
        Encodes the input, samples from the latent distribution, and decodes.

        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width).
                          Expected to be normalized to [-1, 1].
            sample_posterior: If True, samples from the latent distribution.
                              If False, uses the mean of the distribution (deterministic).

        Returns:
            A dictionary containing:
            - 'reconstruction': The reconstructed pixel values tensor.
            - 'latent_dist': The DiagonalGaussianDistribution object from the encoder.
            - 'latents_sampled': The sampled (or mean) latents *before* scaling.
        """
        # Encode
        latent_dist = self.vae.encode(pixel_values).latent_dist

        # Sample or get mean
        if sample_posterior:
            latents = latent_dist.sample()
        else:
            latents = latent_dist.mode() # Or .mean

        # Decode
        # Note: We do NOT scale latents during VAE training itself.
        # Scaling is typically applied *after* sampling for use in diffusion models.
        reconstruction = self.vae.decode(latents).sample

        return {
            "reconstruction": reconstruction,
            "latent_dist": latent_dist,
            "latents_sampled": latents # Return the unscaled latents
        }

    def _capture_activation_hook_fn(self, name: str) -> Callable:
        """
        Returns a hook function that captures the output of a module
        and stores it in _captured_activations.
        """
        def hook(module: torch.nn.Module, input_data: torch.Tensor, output_data: torch.Tensor):
            # We detach and move to CPU immediately to manage GPU memory,
            # especially if many activations are captured or for large models.
            self._captured_activations[name] = output_data.detach().cpu()
            # logger.debug(f"Captured activation for '{name}'. Shape: {output_data.shape}")
        return hook

    def add_hooks(self, layer_names: List[str]):
        """
        Registers forward hooks on specified layers of the VAE.
        Clears existing hooks first to avoid duplicates.

        Args:
            layer_names: A list of string identifiers for the layers (e.g., 'encoder.conv_in').
                         These names must match those from self.vae.named_modules().
        """
        # Clear any previously registered hooks to ensure a clean state
        self.remove_hooks()

        found_any_hook = False
        for name, module in self.vae.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._capture_activation_hook_fn(name))
                self._hook_handles.append(handle)
                logger.info(f"Registered activation hook for VAE layer: '{name}'")
                found_any_hook = True

        if not found_any_hook and layer_names: # Warn only if layers were specified but none found
            print(layer_names)
            logger.warning(f"No hooks registered. Ensure layer names {layer_names} are correct and exist in the VAE.")

    def remove_hooks(self):
        """
        Removes all active forward hooks registered by this wrapper and clears
        the dictionary of captured activations.
        """
        if not self._hook_handles:
            # logger.debug("No hooks to remove.")
            return

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._captured_activations.clear()
        logger.info("Cleared all VAE model hooks and captured activations.")

    def get_captured_activations(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of captured activations.
        Keys are layer names, values are the activation tensors (on CPU, detached).
        """
        return self._captured_activations

    def clear_captured_activations(self):
        """
        Clears the stored captured activations without removing the hooks themselves.
        Useful if you want to capture activations for a new batch.
        """
        self._captured_activations.clear()
        logger.debug("Cleared captured activations from memory.")

    # --- Convenience methods for inference (similar to sdxl_vae.py) ---

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes pixel values into scaled latents for diffusion models.

        Args:
            pixel_values: Input tensor normalized to [-1, 1].

        Returns:
            Scaled latent tensor.
        """
        self.vae.eval() # Ensure eval mode
        latent_dist = self.vae.encode(pixel_values.to(self.vae.device, dtype=self.vae.dtype)).latent_dist
        latents = latent_dist.sample() # Or .mode() for deterministic encoding
        latents = latents * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes scaled latents back into pixel space.

        Args:
            latents: Scaled latent tensor from a diffusion model.

        Returns:
            Decoded pixel tensor normalized to [-1, 1].
        """
        self.vae.eval() # Ensure eval mode
        latents = latents / self.scaling_factor
        image = self.vae.decode(latents.to(self.vae.device, dtype=self.vae.dtype)).sample
        image = image.clamp(-1, 1) # Clamp output
        return image


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example Usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        logger.info("--- Initializing VAE Wrapper ---")
        vae_wrapper = SDXLVAEWrapper(torch_dtype=dtype).to(device)
        vae_wrapper.vae.eval() # Set to eval for inference testing

        logger.info("\n--- Testing Forward Pass ---")
        # Create a dummy input batch (e.g., 1 image, 3 channels, 256x256)
        dummy_input = torch.randn(1, 3, 256, 256, device=device, dtype=dtype) * 0.5 # Approx [-1, 1]
        output_dict = vae_wrapper(dummy_input)
        print(f"Forward pass output keys: {output_dict.keys()}")
        print(f"Reconstruction shape: {output_dict['reconstruction'].shape}")
        print(f"Latent dist type: {type(output_dict['latent_dist'])}")
        print(f"Sampled latents shape: {output_dict['latents_sampled'].shape}")

        logger.info("\n--- Testing Encode/Decode Inference ---")
        # Encode
        latents_scaled = vae_wrapper.encode(dummy_input)
        print(f"Encoded (scaled) latents shape: {latents_scaled.shape}")

        # Decode
        decoded_image = vae_wrapper.decode(latents_scaled)
        print(f"Decoded image shape: {decoded_image.shape}")
        print(f"Decoded image min/max: {decoded_image.min():.4f}, {decoded_image.max():.4f}")

    except Exception as e:
        logger.error(f"An error occurred during example usage: {e}", exc_info=True)
