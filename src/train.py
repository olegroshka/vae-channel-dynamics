# src/train.py
import os
import sys
import argparse
import logging
import math
import warnings
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import wandb
import yaml

# Local imports
from utils.config_utils import load_config
from utils.logging_utils import setup_logging
from utils.plotting_utils import DeadNeuronPlotter # <<< Import the new plotter class
from data_utils import load_and_preprocess_dataset, create_dataloader
from models.sdxl_vae_wrapper import SDXLVAEWrapper
from tracking.monitor import ActivityMonitor
from classification.classifier import RegionClassifier
from intervention.nudger import InterventionHandler

# Setup basic logging configuration first
setup_logging()
logger = get_logger(__name__, log_level="INFO") # Use accelerate logger

# Filter user warnings (e.g., from datasets library)
warnings.filterwarnings("ignore", category=UserWarning)

# target layers for dead neuron analysis (based on user code)
target_layer_classes = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)
target_layer_names = ["decoder.conv_out.weight", "decoder.conv_in.weight"]

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train/Fine-tune SDXL VAE with channel dynamics analysis.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiment.",
    )
    # Add other command-line overrides if needed, e.g.,
    # parser.add_argument("--output_dir", type=str, help="Override output directory.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_config(args.config_path)

    # --- Accelerator Setup ---
    run_name = config.get("run_name", "vae_channel_dynamics_run")
    # *** Get threshold from config, ensuring it's float ***
    try:
        threshold = float(config.get("threshold", 1e-8))
    except (ValueError, TypeError):
        logger.error(f"Invalid threshold format: {config.get('threshold')}. Using default 1e-8.")
        threshold = 1e-8

    output_dir = os.path.join(config.get("output_dir", "./results"), run_name)
    logging_dir = os.path.join(output_dir, "logs")
    logging_config = config.get("logging", {}) # Get logging sub-config
    report_to = logging_config.get("report_to", "tensorboard")
    mixed_precision_config = config.get("training", {}).get("mixed_precision", "no") # Get mixed precision from config

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    # Initialize accelerator *without* wandb initially if we handle it manually
    log_with_accelerate = None if report_to in ["wandb", "all"] else report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision_config, # Use the value from config
        log_with=log_with_accelerate, # Use None if wandb is primary
        project_config=accelerator_project_config,
    )

    # --- Logging Setup ---
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets_log = logging.getLogger("datasets")
        datasets_log.setLevel(logging.WARNING) # Reduce datasets verbosity
        diffusers_log = logging.getLogger("diffusers")
        diffusers_log.setLevel(logging.WARNING) # Reduce diffusers verbosity
        # Suppress matplotlib font warnings if desired
        mpl_logger = logging.getLogger('matplotlib.font_manager')
        mpl_logger.setLevel(logging.WARNING)
    else:
        # Only log errors on non-main processes
        pass


    logger.info(f"Running experiment: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using config file: {args.config_path}")
    # Log the config content
    import json
    logger.info(f"Loaded configuration:\n{json.dumps(config, indent=2)}")


    # --- Seed ---
    if config.get("seed") is not None:
        set_seed(config["seed"])
        logger.info(f"Set random seed to {config['seed']}")

    # --- Create Output Dirs ---
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        # Save config to output dir
        config_save_path = os.path.join(output_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_save_path}")


    # --- Wandb Initialization (if used) ---
    # Extract wandb entity from config
    wandb_entity = logging_config.get("entity", None) # Get entity, default to None
    # Determine if wandb should be used based on config and process rank
    use_wandb = accelerator.is_main_process and report_to in ["wandb", "all"]

    if use_wandb:
        try:
            wandb.init(
                project=config.get("project_name", "vae-channel-dynamics"),
                name=run_name,
                config=config,
                dir=output_dir, # Store wandb files within the run's output dir
                entity=wandb_entity, # <<< Pass the entity (team name or username)
                # mode="disabled" # Uncomment to disable wandb locally
            )
            logger.info(f"Weights & Biases initialized (Entity: {wandb_entity or 'default'}).")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}. Continuing without wandb.")
            use_wandb = False # Disable wandb if init fails
            # If wandb failed, maybe default accelerator logging to tensorboard?
            if accelerator.log_with is None:
                 accelerator.log_with = "tensorboard"
                 logger.warning("Falling back to accelerator logging with tensorboard.")


    accelerator.wait_for_everyone()

    # --- Load Model ---
    model_config = config.get("model", {})
    # *** FIX: Map config string to correct torch dtype ***
    model_torch_dtype = None
    if mixed_precision_config == "fp16":
        model_torch_dtype = torch.float16
    elif mixed_precision_config == "bf16":
        model_torch_dtype = torch.bfloat16
    # If 'no' or other value, model_torch_dtype remains None (defaults to float32)

    vae_wrapper = SDXLVAEWrapper(
        pretrained_model_name_or_path=model_config.get("pretrained_vae_name", "stabilityai/sdxl-vae"),
        torch_dtype=model_torch_dtype # Pass the mapped dtype
    )
    # VAE is often trained/fine-tuned in full precision for stability, even if rest uses mixed
    # vae_wrapper.vae.to(accelerator.device) # Keep VAE on device, maybe in fp32?
    # Or let accelerate handle it:
    # vae_wrapper = accelerator.prepare(vae_wrapper) # But this might put it in mixed precision

    # --- Load Dataset ---
    data_config = config.get("data", {})
    train_dataset = load_and_preprocess_dataset(
        dataset_name=data_config.get("dataset_name"),
        dataset_config_name=data_config.get("dataset_config_name", None), # <<< Pass dataset config name
        image_column=data_config.get("image_column", "image"),
        resolution=data_config.get("resolution", 256),
        max_samples=data_config.get("max_samples", None),
        split="train",
        # cache_dir= # Optional cache dir
    )

    # --- Create DataLoader ---
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=data_config.get("batch_size", 4),
        num_workers=data_config.get("num_workers", 0),
        shuffle=True # Shuffle for training
    )

    # --- Optimizer ---
    training_config = config.get("training", {})
    # *** FIX: Explicitly cast learning rate to float ***
    try:
        learning_rate = float(training_config.get("learning_rate", 1e-5))
    except (ValueError, TypeError):
        logger.error(f"Invalid learning rate format: {training_config.get('learning_rate')}. Using default 1e-5.")
        learning_rate = 1e-5

    optimizer = torch.optim.AdamW(
        vae_wrapper.parameters(), # Train all params by default
        lr=learning_rate, # Use the converted float value
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        weight_decay=training_config.get("adam_weight_decay", 1e-2),
        eps=training_config.get("adam_epsilon", 1e-08),
    )

    # --- LR Scheduler ---
    # Calculate num_update_steps_per_epoch carefully if dataset is streaming or length unknown
    try:
        num_samples = len(train_dataset) if hasattr(train_dataset, "__len__") else None
        if num_samples:
            num_update_steps_per_epoch = math.ceil(num_samples / data_config.get("batch_size", 4) / training_config.get("gradient_accumulation_steps", 1))
            logger.info(f"Calculated num_update_steps_per_epoch: {num_update_steps_per_epoch}")
        else:
             # Estimate or set a large number if dataset length is unknown (streaming)
             # This might affect LR decay schedule if not set appropriately
             num_update_steps_per_epoch = 10000 # Example large number, adjust as needed
             logger.warning(f"Dataset length unknown (streaming?). Estimating num_update_steps_per_epoch as {num_update_steps_per_epoch}. LR decay might be inaccurate.")
    except TypeError:
        num_update_steps_per_epoch = 10000 # Fallback if len() not supported
        logger.warning(f"Could not determine dataset length. Estimating num_update_steps_per_epoch as {num_update_steps_per_epoch}. LR decay might be inaccurate.")

    # Ensure num_train_epochs is treated as int
    num_train_epochs = int(training_config.get("num_train_epochs", 1))
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_warmup_steps = int(training_config.get("lr_warmup_steps", 100)) # Ensure int

    # Simple linear warmup and decay function
    def lr_lambda(current_step: int):
        if current_step < lr_warmup_steps:
            return float(current_step) / float(max(1, lr_warmup_steps))
        # Ensure max_train_steps is not less than lr_warmup_steps for decay calculation
        decay_steps = max(1, max_train_steps - lr_warmup_steps)
        progress = float(current_step - lr_warmup_steps) / float(decay_steps)
        # Ensure progress doesn't exceed 1.0, handle edge case max_train_steps == lr_warmup_steps
        progress = min(1.0, progress)
        return max(0.0, 1.0 - progress)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Prepare with Accelerator ---
    logger.info("Preparing components with Accelerator...")
    vae_wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae_wrapper, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Preparation complete.")

    # --- Initialize Tracking, Classification, Intervention ---
    # Pass relevant config sections
    monitor = ActivityMonitor(vae_wrapper, config.get("tracking", {}).get("target_layers", []), config.get("tracking", {}))
    classifier = RegionClassifier(config.get("classification", {}))
    intervention_handler = InterventionHandler(vae_wrapper, config.get("intervention", {}))


    # --- Training Loop ---
    total_batch_size = data_config.get("batch_size", 4) * accelerator.num_processes * training_config.get("gradient_accumulation_steps", 1)
    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}") # Might be unknown for streaming
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {data_config.get('batch_size', 4)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_config.get('gradient_accumulation_steps', 1)}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    # Ensure kl_weight is float
    try:
        kl_weight = float(training_config.get("kl_weight", 1e-6))
    except (ValueError, TypeError):
        logger.error(f"Invalid kl_weight format: {training_config.get('kl_weight')}. Using default 1e-6.")
        kl_weight = 1e-6

    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    log_interval = logging_config.get("log_interval", 10) # Use logging_config
    save_interval = config.get("saving", {}).get("save_interval", 500)
    checkpoint_dir_prefix = config.get("saving", {}).get("checkpoint_dir_prefix", "chkpt")

    # --- Checkpoint Resumption (Basic Example) ---
    # TODO: Implement more robust checkpoint loading if needed
    # resume_from_checkpoint = None
    # if resume_from_checkpoint: ... load state ...

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )

    percent_history = defaultdict(list) # Initialize dead neuron history tracking
    weights_history = defaultdict(list)
    for epoch in range(first_epoch, num_train_epochs):
        vae_wrapper.train() # Set model to training mode
        train_loss_accum = 0.0
        rec_loss_accum = 0.0
        kl_loss_accum = 0.0
        step_count_in_epoch = 0 # Count steps where loss was calculated in this epoch

        for step, batch in enumerate(train_dataloader):
            # Ensure batch is on the correct device (handled by accelerator.prepare on dataloader)
            pixel_values = batch.get("pixel_values")
            if pixel_values is None:
                 logger.warning(f"Step {step}: Batch did not contain 'pixel_values', skipping.")
                 continue
            if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
                 logger.warning(f"Step {step}: 'pixel_values' is not a valid 4D tensor, skipping batch. Got type: {type(pixel_values)}")
                 continue
            if pixel_values.shape[0] == 0:
                 logger.warning(f"Step {step}: Received empty batch, skipping.")
                 continue

            # VAEs might be sensitive to mixed precision, potentially cast input
            # pixel_values = pixel_values.to(vae_wrapper.vae.dtype) # Let accelerator handle casting

            with accelerator.accumulate(vae_wrapper):
                # Forward pass
                model_output = vae_wrapper(pixel_values, sample_posterior=True)
                reconstruction = model_output["reconstruction"]
                latent_dist = model_output["latent_dist"]

                # Calculate loss
                # Ensure inputs to loss are float32 for stability if using mixed precision
                rec_loss = F.mse_loss(reconstruction.float(), pixel_values.float(), reduction="mean")
                kl_loss = latent_dist.kl().mean() # .kl() should handle precision internally
                total_loss = rec_loss + kl_weight * kl_loss

                # Accumulate loss values for logging (average across processes)
                # Use .detach() to avoid holding onto computation graph
                avg_loss = accelerator.gather(total_loss.detach().repeat(pixel_values.shape[0])).mean()
                avg_rec_loss = accelerator.gather(rec_loss.detach().repeat(pixel_values.shape[0])).mean()
                avg_kl_loss = accelerator.gather(kl_loss.detach().repeat(pixel_values.shape[0])).mean()

                train_loss_accum += avg_loss.item()
                rec_loss_accum += avg_rec_loss.item()
                kl_loss_accum += avg_kl_loss.item()
                step_count_in_epoch += 1


                # Backward pass
                accelerator.backward(total_loss)

                # Gradient clipping and optimizer step
                if accelerator.sync_gradients:
                    if max_grad_norm is not None:
                         # Ensure parameters being clipped are valid
                         params_to_clip = [p for p in vae_wrapper.parameters() if p.requires_grad]
                         if params_to_clip:
                              accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                         else:
                              logger.warning("No parameters requiring gradients found for clipping.")

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # --- Tracking, Classification, Intervention ---
                    # (Run these *after* optimizer step and *before* next forward pass)
                    tracked_data = monitor.step(global_step)
                    classification_results = classifier.classify(tracked_data or {}, global_step)
                    intervention_handler.intervene(classification_results or {}, global_step)
                    # ---------------------------------------------

                    progress_bar.update(1)
                    global_step += 1

                    # --- Logging ---
                    if global_step % log_interval == 0 and step_count_in_epoch > 0:
                        # Calculate average loss since last log
                        # Divide by step_count_in_epoch which tracks how many batches contributed to accumulators
                        avg_step_loss = train_loss_accum / step_count_in_epoch
                        avg_step_rec_loss = rec_loss_accum / step_count_in_epoch
                        avg_step_kl_loss = kl_loss_accum / step_count_in_epoch

                        logs = {
                            "train_loss": avg_step_loss,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "reconstruction_loss": avg_step_rec_loss,
                            "kl_loss": avg_step_kl_loss,
                            "epoch": epoch,
                            "step": global_step, # Log using global_step
                        }

                        # <<< Add this debug log >>>
                        # logger.info(f"Logging metrics at step {global_step}: {logs}") # Keep commented unless debugging logging

                        progress_bar.set_postfix(**{k: f"{v:.4e}" if isinstance(v, float) else v for k, v in logs.items()})
                        try:
                            # *** Conditionally log based on use_wandb ***
                            if use_wandb:
                                if wandb.run is not None:
                                    wandb.log(logs, step=global_step)
                                    # logger.info(f"Explicitly logged metrics to wandb at step {global_step}") # Reduce verbosity
                            elif accelerator.log_with is not None: # Log with accelerator if not using wandb
                                accelerator.log(logs, step=global_step)

                        except Exception as log_e:
                             logger.error(f"Error during logging: {log_e}")


                        # Reset accumulators for next logging interval
                        train_loss_accum = 0.0
                        rec_loss_accum = 0.0
                        kl_loss_accum = 0.0
                        step_count_in_epoch = 0


                    # --- Save Checkpoint ---
                    if global_step % save_interval == 0:
                        if accelerator.is_main_process:
                            chkpt_save_dir = os.path.join(output_dir, f"{checkpoint_dir_prefix}-{global_step}")
                            os.makedirs(chkpt_save_dir, exist_ok=True)
                            logger.info(f"Saving checkpoint state to {chkpt_save_dir}")
                            try:
                                accelerator.save_state(chkpt_save_dir)
                                # Optionally save the unwrapped model separately if needed
                                # unwrapped_model = accelerator.unwrap_model(vae_wrapper)
                                # unwrapped_model.vae.save_pretrained(os.path.join(chkpt_save_dir, "vae_unwrapped"))
                            except Exception as save_e:
                                 logger.error(f"Error saving checkpoint state: {save_e}")


            if global_step >= max_train_steps:
                break # Exit inner loop

        # --- End of Epoch ---
        logger.info(f"Epoch {epoch} completed.")

        # --- Collect Dead Neuron Stats (at end of epoch) ---
        if accelerator.is_main_process: # Only main process needs to calculate/store this
            epoch_dead_neuron_stats = {}
            current_model = accelerator.unwrap_model(vae_wrapper) # Get the unwrapped model
            for name, param in current_model.vae.named_parameters():
                if name in target_layer_names:
                    weights_history[name].append(param.detach().cpu().numpy())
                if "weight" in name and param.requires_grad: # Focus on weights
                    module_path = ".".join(name.split(".")[:-1])
                    try:
                        # Check if the module associated with the weight is of a target type
                        module = current_model.vae.get_submodule(module_path)
                        if isinstance(module, target_layer_classes):
                            total_elements = param.numel()
                            if total_elements > 0: # Avoid division by zero
                                small_values = (param.abs() < threshold).sum().item()
                                percentage = (small_values / total_elements) * 100
                                # Only store if percentage is non-negligible to avoid clutter? Optional.
                                # if percentage > 1e-4:
                                percent_history[name].append(percentage) # Store history per layer name
                                epoch_dead_neuron_stats[f"dead_neurons/{name}_perc"] = percentage # Log current epoch stat
                                logger.debug(f"Epoch {epoch} - {name}: {percentage:.4f}% values < {threshold}")
                            else:
                                logger.debug(f"Epoch {epoch} - {name}: Skipping, 0 elements.")

                    except AttributeError: # Handle cases where get_submodule fails
                        logger.debug(f"Could not get submodule for {name}, skipping dead neuron check.")
                        continue
                    except Exception as e: # Catch other potential errors
                         logger.error(f"Error checking dead neurons for {name}: {e}")

            # --- Log Dead Neuron Stats to Wandb/Tensorboard ---
            if epoch_dead_neuron_stats: # Log if any stats were collected
                try:
                    step_to_log_epoch_stats = global_step # Log epoch stats at the current global step
                    if use_wandb:
                        if wandb.run is not None:
                            wandb.log(epoch_dead_neuron_stats, step=step_to_log_epoch_stats)
                            logger.info(f"Logged dead neuron stats to wandb for epoch {epoch} at step {step_to_log_epoch_stats}")
                    elif accelerator.log_with is not None:
                        accelerator.log(epoch_dead_neuron_stats, step=step_to_log_epoch_stats)
                        logger.info(f"Logged dead neuron stats via accelerator for epoch {epoch} at step {step_to_log_epoch_stats}")
                except Exception as log_e:
                    logger.error(f"Error logging dead neuron stats for epoch {epoch}: {log_e}")
        # ------------------------------------------------------

        if global_step >= max_train_steps:
             logger.info("Reached max_train_steps. Exiting training.")
             break # Exit outer loop


    # --- End of Training ---
    accelerator.wait_for_everyone()
    logger.info("Training finished.")

    # --- Add a small delay before finishing wandb ---
    if use_wandb:
        logger.info("Pausing for 10 seconds before finishing wandb run...") # Increased sleep time
        time.sleep(10)


    # --- Save Final Model ---
    if accelerator.is_main_process:
        final_model_save_path = os.path.join(output_dir, "final_model")
        logger.info(f"Saving final model state to {final_model_save_path}")
        try:
            accelerator.save_state(final_model_save_path) # Save optimizer, scheduler etc.

            logger.info(f"Saving final unwrapped VAE model to {final_model_save_path}/vae")
            unwrapped_model = accelerator.unwrap_model(vae_wrapper)
            unwrapped_model.vae.save_pretrained(os.path.join(final_model_save_path, "vae"))
        except Exception as final_save_e:
            logger.error(f"Error saving final model/state: {final_save_e}")

        # --- Plot and Save Dead Neuron History ---
        plot_path = os.path.join(output_dir, "dead_neuron_percentage_history.png")
        csv_path = os.path.join(output_dir, "dead_neuron_percentage_history.csv")
        plotter = DeadNeuronPlotter(threshold=threshold) # Use default top_n=10 or configure via config
        plotter.plot_history(percent_history, plot_path, csv_path)
        plotter.plot_dead_over_epoch(weights_history, output_dir)
        plotter.plot_heatmap(weights_history, output_dir)
        logger.info(f"Saved dead neuron history plot to {plot_path}")
        # -----------------------------------------

        # --- End Wandb Run ---
        if use_wandb and wandb.run is not None:
            try:
                 # Log the plot as an artifact/image to wandb
                 if os.path.exists(plot_path): # Check if plot was actually saved
                     wandb.log({"dead_neuron_plot": wandb.Image(plot_path)})
                     logger.info("Logged dead neuron plot to wandb.")
                 else:
                      logger.warning("Dead neuron plot file not found, cannot log to wandb.")
                 wandb.finish()
                 logger.info("Weights & Biases run finished.")
            except Exception as wandb_e:
                 logger.error(f"Error finishing wandb run or logging plot: {wandb_e}")


    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled exception occurred in main: {e}", exc_info=True)
        sys.exit(1) # Exit with error code

