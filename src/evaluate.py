# src/evaluate.py
import argparse
import os
import sys

import torch
import torch.nn.functional as F  # <<< Added missing import for F.mse_loss
import torchvision.transforms as T  # <<< Added missing import
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from data_utils import load_and_preprocess_dataset, \
    create_dataloader  # Removed get_transform as it's defined locally now in data_utils
from models.sdxl_vae_wrapper import SDXLVAEWrapper
# Local imports
from utils.config_utils import load_config
from utils.logging_utils import setup_logging

from analysis.logit_lens import VAELogitLens

# Setup basic logging
setup_logging()
logger = get_logger(__name__, log_level="INFO")

def parse_args():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SDXL VAE model.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file used for training (or a specific eval config).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory (containing the 'vae' subdirectory).",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split to use for evaluation (e.g., 'test', 'validation').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (e.g., reconstructed images). Defaults to checkpoint_path/eval_results.",
    )
    parser.add_argument(
        "--num_samples_to_save",
        type=int,
        default=16,
        help="Number of reconstructed image samples to save.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None, # Use config batch size by default
        help="Override evaluation batch size.",
    )
    parser.add_argument(
        "--enable_logit_lens",
        default=True,
        help="Enable Logit Lens analysis during evaluation.",
    )
    parser.add_argument(
        "--logit_lens_layers",
        type=str,
        nargs="+",
        default=["encoder.down_blocks.0.resnets.0.norm1", "encoder.down_blocks.1.resnets.0.conv_shortcut"],
        help="Space-separated list of layer names to apply Logit Lens to (e.g., 'encoder.conv1' 'encoder.conv2')."
             "Refer to your VAE's module names (e.g., by printing model.named_modules())."
             "For the default AutoencoderKL, common encoder layers are 'encoder.conv_in', 'encoder.down_blocks.0.resnets.0.norm1', etc.",
    )
    parser.add_argument(
        "--logit_lens_num_samples",
        type=int,
        default=1,
        help="Number of batch samples to project for Logit Lens visualization.",
    )
    parser.add_argument(
        "--logit_lens_projection_type",
        type=str,
        default="mini_decoder_single_channel",
        choices=["mini_decoder_single_channel", "mini_decoder_full_map"],
        help="Type of projection for Logit Lens. 'mini_decoder_single_channel' or 'mini_decoder_full_map'.",
    )
    parser.add_argument(
        "--logit_lens_mini_decoder_input_channels",
        type=int,
        default=None, # Will try to infer if not provided
        help="Input channels for the Logit Lens mini-decoder. Crucial if using 'mini_decoder_full_map'."
             "If not provided, defaults to 1 for 'mini_decoder_single_channel'."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_config(args.config_path)

    # --- Accelerator Setup (minimal for evaluation) ---
    # Determine mixed precision from config, default to 'no' if not specified
    mixed_precision_setting = config.get("training", {}).get("mixed_precision", "no")
    accelerator = Accelerator(mixed_precision=mixed_precision_setting)
    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Using mixed precision: {mixed_precision_setting}")


    # --- Output Dir ---
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, f"eval_results_{args.eval_split}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {args.output_dir}")

    # --- Load Model ---
    model_load_path = os.path.join(args.checkpoint_path, "vae") # Expecting unwrapped VAE here
    if not os.path.isdir(model_load_path):
         logger.error(f"VAE model directory not found at: {model_load_path}")
         logger.error(f"Ensure the checkpoint directory contains a 'vae' subdirectory with the saved AutoencoderKL model.")
         return

    logger.info(f"Loading VAE model from: {model_load_path}")
    try:
        # Determine dtype based on accelerator's mixed precision setting
        model_dtype = getattr(torch, accelerator.mixed_precision) if accelerator.mixed_precision != "no" else None
        vae_wrapper = SDXLVAEWrapper(
            pretrained_model_name_or_path=model_load_path,
            torch_dtype=model_dtype # Pass the determined dtype
        )
        vae_wrapper.vae.eval() # Set to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    logit_lens_analyzer = None
    if args.enable_logit_lens:
        logger.info("Logit Lens analysis enabled.")
        # If mini_decoder_input_channels is not specified, infer it for single_channel
        # For full_map, it MUST be specified.
        ll_mini_decoder_input_channels = args.logit_lens_mini_decoder_input_channels
        if args.logit_lens_projection_type == "mini_decoder_single_channel" and ll_mini_decoder_input_channels is None:
            ll_mini_decoder_input_channels = 1 # Single channel projection expects 1 input channel
            logger.info(f"Defaulting mini_decoder_input_channels to 1 for '{args.logit_lens_projection_type}'")
        elif args.logit_lens_projection_type == "mini_decoder_full_map" and ll_mini_decoder_input_channels is None:
            logger.error(
                f"For 'mini_decoder_full_map' projection, --logit_lens_mini_decoder_input_channels must be specified "
                f"(it should match the 'Channels' dimension of the activation map)."
            )
            sys.exit(1) # Exit if critical config is missing

        logit_lens_config = {
            "visualization_output_subdir": "logit_lens_visualizations",
            "default_num_channels_to_viz": 4, # Used by channel visualization
            "default_num_batch_samples_to_viz": args.logit_lens_num_samples, # For channel viz
            "mini_decoder_input_channels": ll_mini_decoder_input_channels
        }
        # Pass the actual VAE model (AutoencoderKL) to VAELogitLens
        logit_lens_analyzer = VAELogitLens(
            model_for_lens=vae_wrapper.vae, # Pass the underlying AutoencoderKL
            logit_lens_config=logit_lens_config,
            main_experiment_output_dir=args.output_dir # Use eval output dir
        )

    # --- Load Dataset ---
    data_config = config.get("data", {})
    eval_batch_size = args.batch_size if args.batch_size is not None else data_config.get("batch_size", 4)
    eval_dataset = load_and_preprocess_dataset(
        dataset_name=data_config.get("dataset_name"),
        image_column=data_config.get("image_column", "image"),
        resolution=data_config.get("resolution", 256),
        max_samples=None, # Evaluate on the full split
        split=args.eval_split,
    )

    # --- Create DataLoader ---
    eval_dataloader = create_dataloader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=data_config.get("num_workers", 0),
        shuffle=False # No shuffling for evaluation
    )

    # --- Prepare with Accelerator ---
    # Only prepare the dataloader, model is loaded directly to the device if needed
    vae_wrapper = vae_wrapper.to(accelerator.device) # Manually move model
    eval_dataloader = accelerator.prepare(eval_dataloader) # Prepare dataloader

    # --- Evaluation Loop ---
    total_mse = 0.0
    total_kl = 0.0 # KL might not be meaningful if just evaluating reconstruction
    num_batches = 0
    samples_saved = 0

    # Inverse transform to get PIL images back
    postprocess = T.Compose([
        T.Normalize([-1.0], [2.0]), # Maps [-1.0, 1.0] back to [0.0, 1.0]
        T.ToPILImage()
    ])

    logger.info(f"Starting evaluation on '{args.eval_split}' split...")
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, desc="Evaluating")

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if step == 0:
                vae_wrapper.add_hooks(args.logit_lens_layers)

            pixel_values = batch.get("pixel_values")
            if pixel_values is None:
                continue
            # Ensure pixel_values are on the correct device (model device)
            pixel_values = pixel_values.to(accelerator.device)

            # Forward pass through the VAE wrapper
            # Cast input to model's dtype if using mixed precision
            input_dtype = vae_wrapper.vae.dtype
            model_output = vae_wrapper(pixel_values.to(input_dtype), sample_posterior=False) # Use deterministic mode for reconstruction eval
            reconstruction = model_output["reconstruction"]
            latent_dist = model_output["latent_dist"]

            # Calculate metrics (on CPU after gathering)
            # Gather tensors before moving to CPU
            gathered_reconstruction = accelerator.gather(reconstruction.contiguous())
            gathered_pixel_values = accelerator.gather(pixel_values.contiguous()) # Gather original inputs too
            gathered_kl = accelerator.gather(latent_dist.kl().contiguous()) # Gather KL divergence

            # Ensure tensors are on CPU for metric calculation
            # Use float32 for metric calculation for stability
            mse = F.mse_loss(gathered_reconstruction.float().cpu(), gathered_pixel_values.float().cpu(), reduction="mean")
            kl = gathered_kl.mean().float().cpu() # Ensure KL is float and on CPU

            total_mse += mse.item()
            total_kl += kl.item()
            num_batches += 1

            # Save some sample images (only on main process)
            if accelerator.is_main_process and samples_saved < args.num_samples_to_save:
                num_to_save_this_batch = min(args.num_samples_to_save - samples_saved, gathered_pixel_values.size(0))
                for i in range(num_to_save_this_batch):
                    # Ensure tensors are float and on CPU before postprocessing
                    original_pil = postprocess(gathered_pixel_values[i].float().cpu())
                    reconstructed_pil = postprocess(gathered_reconstruction[i].float().cpu())

                    try:
                        original_pil.save(os.path.join(args.output_dir, f"sample_{samples_saved}_orig.png"))
                        reconstructed_pil.save(os.path.join(args.output_dir, f"sample_{samples_saved}_recon.png"))
                        samples_saved += 1
                    except Exception as img_save_e:
                         logger.error(f"Error saving image sample {samples_saved}: {img_save_e}")


            progress_bar.update(1)
            # Avoid division by zero if num_batches is 0 (shouldn't happen here)
            current_avg_mse = total_mse / num_batches if num_batches > 0 else 0
            current_avg_kl = total_kl / num_batches if num_batches > 0 else 0
            progress_bar.set_postfix(MSE=f"{current_avg_mse:.4e}", KL=f"{current_avg_kl:.4e}")

            if step == 0:
                activations = vae_wrapper.get_captured_activations()
                for layer, activation in activations.items():
                    for i in range(min(activation.shape[0], 10)):
                        out = activation[i].squeeze(0)
                        out.clamp(min=out.min(), max=out.max())
                        out_grid = out.unsqueeze(1)
                        grid_img = make_grid(out_grid, nrow=8, normalize=True, padding=2)
                        output_pil = T.ToPILImage()(grid_img)
                        output_pil.save(f"{args.output_dir}/out_{i}.png")
                        print(f"Saved to {args.output_dir}/out_{i}.png")

                if args.enable_logit_lens and logit_lens_analyzer is not None:
                    if accelerator.is_local_main_process:
                        logit_lens_analyzer.run_logit_lens_with_activations(
                            global_step=0,
                            layers_to_analyze=args.logit_lens_layers,
                            num_batch_samples_to_viz=args.logit_lens_num_samples,
                            projection_type=args.logit_lens_projection_type,
                            activations_to_process=activations,
                        )

                vae_wrapper.remove_hooks()


    # --- Final Metrics ---
    # Avoid division by zero if dataset was empty
    avg_mse = total_mse / num_batches if num_batches > 0 else 0
    avg_kl = total_kl / num_batches if num_batches > 0 else 0


    logger.info("***** Evaluation Results *****")
    logger.info(f"  Dataset split: {args.eval_split}")
    logger.info(f"  Number of batches processed: {num_batches}")
    logger.info(f"  Average MSE Loss: {avg_mse:.6f}")
    logger.info(f"  Average KL Divergence: {avg_kl:.6f}") # Note: KL depends on sampling strategy used during eval
    logger.info(f"  Saved {samples_saved} image samples to {args.output_dir}")

    # Save metrics to a file
    if accelerator.is_main_process:
        metrics_path = os.path.join(args.output_dir, "eval_metrics.txt")
        try:
            with open(metrics_path, "w") as f:
                f.write(f"Evaluation Split: {args.eval_split}\n")
                f.write(f"Checkpoint Path: {args.checkpoint_path}\n")
                f.write(f"Number of Batches: {num_batches}\n")
                f.write(f"Average MSE: {avg_mse}\n")
                f.write(f"Average KL: {avg_kl}\n")
            logger.info(f"Evaluation metrics saved to {metrics_path}")
        except Exception as metrics_save_e:
            logger.error(f"Error saving metrics file: {metrics_save_e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled exception occurred during evaluation: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
