# src/evaluate.py
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image  # Changed from make_grid in my previous version
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm

# --- New Imports for PSNR & SSIM ---
try:
    import torchmetrics

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    # Fallback if torchmetrics is not installed
    # You could implement basic PSNR/SSIM manually here if needed, or just skip.
    # For now, we'll just set a flag.
    TORCHMETRICS_AVAILABLE = False
# --- End New Imports ---

from data_utils import load_and_preprocess_dataset, create_dataloader
from models.sdxl_vae_wrapper import SDXLVAEWrapper
from utils.config_utils import load_config
from utils.logging_utils import setup_logging
from analysis.logit_lens import VAELogitLens  # Keep user's import

setup_logging()
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SDXL VAE model.")
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the YAML configuration file used for training (or a specific eval config).")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained model checkpoint directory (containing the 'vae' subdirectory).")
    parser.add_argument(
        "--eval_split", type=str, default="test",
        help="Dataset split to use for evaluation (e.g., 'test', 'validation').")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save evaluation results. Defaults to checkpoint_path/eval_results_<split>.")
    parser.add_argument(
        "--num_samples_to_save", type=int, default=16,
        help="Number of reconstructed image samples to save.")
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override evaluation batch size.")
    parser.add_argument(
        "--enable_logit_lens", default=True, type=lambda x: (str(x).lower() == 'true'),  # Handle boolean from string
        help="Enable Logit Lens analysis during evaluation (True/False).")
    parser.add_argument(
        "--logit_lens_layers", type=str, nargs="+",
        default=["encoder.down_blocks.0.resnets.0.norm1", "encoder.down_blocks.1.resnets.0.conv_shortcut"],
        help="Space-separated list of layer names for Logit Lens.")
    parser.add_argument(
        "--logit_lens_num_samples", type=int, default=1,
        help="Number of batch samples for Logit Lens.")
    parser.add_argument(
        "--logit_lens_projection_type", type=str, default="mini_decoder_single_channel",
        choices=["mini_decoder_single_channel", "mini_decoder_full_map"],
        help="Projection type for Logit Lens.")
    parser.add_argument(
        "--logit_lens_mini_decoder_input_channels", type=int, default=None,
        help="Input channels for Logit Lens mini-decoder.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_config(args.config_path)

    mixed_precision_setting = config.get("training", {}).get("mixed_precision", "no")
    accelerator = Accelerator(mixed_precision=mixed_precision_setting)
    logger.info(f"Using device: {accelerator.device}, Mixed precision: {mixed_precision_setting}")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, f"eval_results_{args.eval_split}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {args.output_dir}")

    model_load_path = os.path.join(args.checkpoint_path, "vae")
    if not os.path.isdir(model_load_path):
        logger.error(f"VAE model directory not found at: {model_load_path}")
        sys.exit(1)

    logger.info(f"Loading VAE model from: {model_load_path}")
    try:
        model_dtype = getattr(torch, accelerator.mixed_precision) if accelerator.mixed_precision != "no" else None
        vae_wrapper = SDXLVAEWrapper(
            pretrained_model_name_or_path=model_load_path,
            torch_dtype=model_dtype
        )
        vae_wrapper.vae.eval()  # Ensure VAE is in eval mode
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    logit_lens_analyzer = None
    if args.enable_logit_lens:
        logger.info("Logit Lens analysis enabled.")
        ll_mini_decoder_input_channels = args.logit_lens_mini_decoder_input_channels
        if args.logit_lens_projection_type == "mini_decoder_single_channel" and ll_mini_decoder_input_channels is None:
            ll_mini_decoder_input_channels = 1
        elif args.logit_lens_projection_type == "mini_decoder_full_map" and ll_mini_decoder_input_channels is None:
            logger.error("For 'mini_decoder_full_map', --logit_lens_mini_decoder_input_channels must be specified.")
            sys.exit(1)

        # Ensure logit_lens_config is sourced from the main config if available, or use args
        ll_config_from_main = config.get("logit_lens", {})
        logit_lens_config_eval = {
            "visualization_output_subdir": ll_config_from_main.get("visualization_output_subdir",
                                                                   "logit_lens_visualizations_eval"),
            "default_num_channels_to_viz": ll_config_from_main.get("num_channels_to_viz", 4),
            "default_num_batch_samples_to_viz": args.logit_lens_num_samples,
            "mini_decoder_input_channels": ll_mini_decoder_input_channels,
            "colormap": ll_config_from_main.get("colormap", "viridis"),  # Use colormap from config
            "run_mini_decoder_projection": ll_config_from_main.get("run_mini_decoder_projection", True)
            # Use from config
        }
        logit_lens_analyzer = VAELogitLens(
            model_for_lens=vae_wrapper.vae,
            logit_lens_config=logit_lens_config_eval,
            main_experiment_output_dir=args.output_dir
        )

    data_config = config.get("data", {})
    eval_batch_size = args.batch_size if args.batch_size is not None else data_config.get("validation_batch_size",
                                                                                          data_config.get("batch_size",
                                                                                                          4))

    # Determine dataset parameters for evaluation
    # Prefer validation-specific settings from config if they exist for the eval_split
    if args.eval_split == data_config.get("validation_split_name", "validation"):
        dataset_name_eval = data_config.get("validation_dataset_name", data_config.get("dataset_name"))
        dataset_config_name_eval = data_config.get("validation_dataset_config_name",
                                                   data_config.get("dataset_config_name"))
        max_samples_eval = data_config.get("validation_max_samples", None)
    else:  # For "test" split or other splits, use general dataset settings
        dataset_name_eval = data_config.get("dataset_name")
        dataset_config_name_eval = data_config.get("dataset_config_name")
        max_samples_eval = None  # Evaluate on full test split usually

    image_column_eval = data_config.get("image_column", "image")
    resolution_eval = data_config.get("resolution", 256)

    eval_dataset = load_and_preprocess_dataset(
        dataset_name=dataset_name_eval,
        dataset_config_name=dataset_config_name_eval,
        image_column=image_column_eval,
        resolution=resolution_eval,
        max_samples=max_samples_eval,
        split=args.eval_split,
    )
    eval_dataloader = create_dataloader(
        eval_dataset, batch_size=eval_batch_size,
        num_workers=data_config.get("num_workers", 0), shuffle=False
    )

    prepared_vae_wrapper = vae_wrapper.to(accelerator.device)  # Manually move model
    prepared_eval_dataloader = accelerator.prepare(eval_dataloader)

    total_mse, total_kl, num_eval_samples = 0.0, 0.0, 0  # Changed num_batches to num_eval_samples
    samples_saved_count = 0  # Renamed from samples_saved

    psnr_metric, ssim_metric = None, None
    if TORCHMETRICS_AVAILABLE:
        try:
            psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
            ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(
                data_range=1.0,
                gaussian_kernel=True,
                sigma=1.5, kernel_size=11
            ).to(accelerator.device)  # Added SSIM params
            logger.info("PSNR and SSIM metrics initialized (torchmetrics).")
        except Exception as e:
            logger.error(f"Failed to initialize torchmetrics: {e}. PSNR/SSIM will not be calculated.")
            psnr_metric, ssim_metric = None, None
    else:
        logger.warning("Torchmetrics not available. PSNR and SSIM will not be calculated.")

    postprocess_to_pil = T.Compose([
        T.Normalize([-1.0], [2.0]),
        T.ToPILImage()
    ])

    def normalize_for_metrics_and_clamp(tensor_batch):
        # Input: [-1, 1], Output: [0, 1] clamped
        return torch.clamp((tensor_batch + 1.0) / 2.0, 0.0, 1.0)

    logger.info(f"Starting evaluation on '{args.eval_split}' split...")
    progress_bar = tqdm(total=len(prepared_eval_dataloader), disable=not accelerator.is_local_main_process,
                        desc="Evaluating")

    with torch.no_grad():
        for step, batch in enumerate(prepared_eval_dataloader):
            # LogitLens hooks added only for the first batch in user's original script
            if step == 0 and args.enable_logit_lens and logit_lens_analyzer:
                logger.info(f"Adding LogitLens hooks for layers: {args.logit_lens_layers}")
                prepared_vae_wrapper.add_hooks(args.logit_lens_layers)

            pixel_values = batch.get("pixel_values")
            if pixel_values is None: continue

            current_batch_size = pixel_values.shape[0]
            input_dtype = prepared_vae_wrapper.vae.dtype
            model_output = prepared_vae_wrapper(pixel_values.to(input_dtype), sample_posterior=False)
            reconstruction = model_output["reconstruction"]
            latent_dist = model_output["latent_dist"]

            # Gather for consistent metric calculation across devices
            # These are in model output range (e.g., [-1, 1])
            gathered_reconstruction = accelerator.gather(reconstruction.contiguous())
            gathered_pixel_values = accelerator.gather(pixel_values.contiguous())

            # KL divergence per item, then mean
            kl_per_item = latent_dist.kl()
            gathered_kl_per_item = accelerator.gather(kl_per_item.contiguous())
            kl_for_batch_mean = gathered_kl_per_item.mean()  # This is already an average for the gathered batch

            # MSE (calculated on CPU for stability, using gathered tensors)
            mse_for_batch_mean = F.mse_loss(
                gathered_reconstruction.float().cpu(),
                gathered_pixel_values.float().cpu(),
                reduction="mean"
            )

            # Accumulate weighted sums for overall average
            total_mse += mse_for_batch_mean.item() * gathered_pixel_values.shape[0]
            total_kl += kl_for_batch_mean.item() * gathered_pixel_values.shape[0]
            num_eval_samples += gathered_pixel_values.shape[0]

            # PSNR & SSIM
            if psnr_metric is not None and ssim_metric is not None:
                # Normalize to [0, 1] and ensure on correct device
                originals_0_1 = normalize_for_metrics_and_clamp(gathered_pixel_values).to(accelerator.device)
                reconstructions_0_1 = normalize_for_metrics_and_clamp(gathered_reconstruction).to(accelerator.device)

                psnr_metric.update(reconstructions_0_1, originals_0_1)
                ssim_metric.update(reconstructions_0_1, originals_0_1)

            if accelerator.is_main_process and samples_saved_count < args.num_samples_to_save:
                num_to_save = min(args.num_samples_to_save - samples_saved_count, gathered_pixel_values.size(0))
                for i in range(num_to_save):
                    original_pil = postprocess_to_pil(gathered_pixel_values[i].float().cpu())
                    reconstructed_pil = postprocess_to_pil(gathered_reconstruction[i].float().cpu())
                    try:
                        original_pil.save(os.path.join(args.output_dir, f"sample_{samples_saved_count}_orig.png"))
                        reconstructed_pil.save(os.path.join(args.output_dir, f"sample_{samples_saved_count}_recon.png"))
                        samples_saved_count += 1
                    except Exception as img_e:
                        logger.error(f"Error saving sample {samples_saved_count}: {img_e}")

            progress_bar.update(1)
            current_avg_mse_display = total_mse / num_eval_samples if num_eval_samples > 0 else 0
            current_avg_kl_display = total_kl / num_eval_samples if num_eval_samples > 0 else 0
            progress_bar.set_postfix(MSE=f"{current_avg_mse_display:.4e}", KL=f"{current_avg_kl_display:.4e}")

            # LogitLens analysis on the first batch's activations
            if step == 0 and args.enable_logit_lens and logit_lens_analyzer:
                logger.info("Running LogitLens on first batch activations...")
                activations = prepared_vae_wrapper.get_captured_activations()  # Get from wrapper
                if accelerator.is_local_main_process:  # LogitLens usually saves files, so main process
                    logit_lens_analyzer.run_logit_lens_with_activations(
                        global_step=0,  # Using 0 as it's eval time
                        # Pass relevant args from logit_lens_config_eval or args
                        layers_to_analyze=args.logit_lens_layers,
                        num_batch_samples_to_viz=args.logit_lens_num_samples,
                        projection_type=args.logit_lens_projection_type,
                        activations_to_process=activations)
                prepared_vae_wrapper.remove_hooks()  # Remove hooks after first batch
                logger.info("LogitLens hooks removed.")

    avg_mse = total_mse / num_eval_samples if num_eval_samples > 0 else 0
    avg_kl = total_kl / num_eval_samples if num_eval_samples > 0 else 0

    final_psnr, final_ssim = float('nan'), float('nan')  # Default to NaN
    if psnr_metric is not None and ssim_metric is not None:
        try:
            final_psnr = psnr_metric.compute().item()
            final_ssim = ssim_metric.compute().item()
        except Exception as e:
            logger.error(f"Error computing final PSNR/SSIM: {e}")

    logger.info("***** Evaluation Results *****")
    logger.info(f"  Dataset split: {args.eval_split}")
    logger.info(f"  Number of samples processed: {num_eval_samples}")
    logger.info(f"  Average MSE Loss: {avg_mse:.6f}")
    logger.info(f"  Average KL Divergence: {avg_kl:.6f}")
    if TORCHMETRICS_AVAILABLE:
        logger.info(f"  Average PSNR: {final_psnr:.4f} dB")
        logger.info(f"  Average SSIM: {final_ssim:.4f}")
    logger.info(f"  Saved {samples_saved_count} image samples to {args.output_dir}")

    if accelerator.is_main_process:
        metrics_path = os.path.join(args.output_dir, "eval_metrics.txt")
        try:
            with open(metrics_path, "w") as f:
                f.write(f"Evaluation Split: {args.eval_split}\n")
                f.write(f"Checkpoint Path: {args.checkpoint_path}\n")
                f.write(f"Number of Samples Processed: {num_eval_samples}\n")  # Changed from Batches
                f.write(f"Average MSE: {avg_mse}\n")
                f.write(f"Average KL: {avg_kl}\n")
                if TORCHMETRICS_AVAILABLE:
                    f.write(f"Average PSNR: {final_psnr}\n")
                    f.write(f"Average SSIM: {final_ssim}\n")
            logger.info(f"Evaluation metrics saved to {metrics_path}")
        except Exception as metrics_save_e:
            logger.error(f"Error saving metrics file: {metrics_save_e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled exception occurred during evaluation: {e}", exc_info=True)
        sys.exit(1)
