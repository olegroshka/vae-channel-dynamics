# src/models/sdxl_vae_wrapper.py
import logging
import torch
from diffusers import AutoencoderKL
from typing import Optional, Union

logger = logging.getLogger(__name__)

class SDXLVAEWrapper(torch.nn.Module):
    """
    A wrapper around the Hugging Face AutoencoderKL model (specifically for SDXL VAE)
    to facilitate loading, potential modifications, and integration into the training loop.
    """
    def __init__(self,
                 pretrained_model_name_or_path: str = "stabilityai/sdxl-vae",
                 torch_dtype: Optional[Union[str, torch.dtype]] = None):
        """
        Initializes the VAE wrapper.

        Args:
            pretrained_model_name_or_path: The Hugging Face model ID or local path
                                           to the pre-trained VAE.
            torch_dtype: Optional torch dtype to load the model in (e.g., torch.float16).
        """
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.torch_dtype = torch_dtype
        self.vae = self._load_vae()
        self.scaling_factor = self.vae.config.scaling_factor

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

        # Check if reconstruction from forward pass matches decode(encode(input))
        # Note: Will not match exactly if sample_posterior=True due to sampling randomness
        output_dict_deterministic = vae_wrapper(dummy_input, sample_posterior=False)
        latents_mean = output_dict_deterministic['latent_dist'].mode() # Get mean latents
        reconstruction_from_mean = vae_wrapper.vae.decode(latents_mean).sample.clamp(-1, 1)

        # Compare reconstruction_from_mean with output_dict_deterministic['reconstruction']
        # They should be very close
        diff = torch.abs(reconstruction_from_mean - output_dict_deterministic['reconstruction']).mean()
        print(f"Difference between forward(mean) and decode(mean): {diff.item():.6f}")


    except Exception as e:
        logger.error(f"An error occurred during example usage: {e}", exc_info=True)

