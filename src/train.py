# src/train.py
import os
import sys
import argparse
import logging  # Keep standard logging import
import math
import warnings
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger  # Use Accelerator's get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import wandb
import yaml
import pandas as pd

# Local imports
from utils.config_utils import load_config
from utils.logging_utils import setup_logging  # Your custom setup
from utils.plotting_utils import DeadNeuronPlotter
from data_utils import load_and_preprocess_dataset, create_dataloader
from models.sdxl_vae_wrapper import SDXLVAEWrapper
from tracking.monitor import ActivityMonitor
from tracking.deadneuron import DeadNeuronTracker
from classification.classifier import RegionClassifier  # Ensure this import is correct
from intervention.nudger import InterventionHandler
from analysis.logit_lens import VAELogitLens

# Setup logging using your utility first, then get accelerator's logger
# This ensures your basic config is set before accelerator might add its handlers
setup_logging()
logger = get_logger(__name__, log_level="INFO")  # Get logger AFTER setup_logging
warnings.filterwarnings("ignore", category=UserWarning)

target_layer_classes = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear, torch.nn.GroupNorm)
# Example specific target for raw weight tracking if needed, otherwise DeadNeuronTracker relies on target_layer_classes
target_layer_names_for_dead_neuron_perc = []  # Can be populated from config if a specific named param is always desired


def parse_args():
    parser = argparse.ArgumentParser(description="Train/Fine-tune SDXL VAE with channel dynamics analysis.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiment.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_config(args.config_path)

    run_name = config.get("run_name", "vae_channel_dynamics_run")
    try:
        # Threshold for DeadNeuronTracker (weights)
        threshold_dn = float(config.get("threshold", 1e-8))
    except (ValueError, TypeError):
        logger.error(f"Invalid threshold format for DeadNeuronTracker: {config.get('threshold')}. Using default 1e-8.")
        threshold_dn = 1e-8

    try:
        mean_percentage_dn = float(config.get("mean_percentage", .01))
    except (ValueError, TypeError):
        logger.error(
            f"Invalid mean_percentage format for DeadNeuronTracker: {config.get('mean_percentage')}. Using default .01")
        mean_percentage_dn = .01

    dead_type_dn = config.get("dead_type", "threshold")

    output_dir = os.path.join(config.get("output_dir", "./results"), run_name)
    logging_dir = os.path.join(output_dir, "logs")  # For accelerator logs

    # Setup logging (again) here if you want to ensure file logging per run is specific
    # Or rely on the initial setup_logging() if global logging is sufficient
    # For run-specific file logs:
    # log_file_path = os.path.join(output_dir, "training_run.log")
    # setup_logging(log_file=log_file_path, log_level=logging.INFO) # Overwrites previous basicConfig

    logging_config = config.get("logging", {})
    report_to = logging_config.get("report_to", "tensorboard")
    mixed_precision_config = config.get("training", {}).get("mixed_precision", "no")

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    # Determine log_with for accelerator based on report_to
    if report_to in ["wandb", "all"]:
        log_with_accelerate = None  # wandb handled separately
    elif report_to == "none":
        log_with_accelerate = None
    else:  # tensorboard
        log_with_accelerate = report_to

    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision_config,
        log_with=log_with_accelerate,  # Use 'tensorboard' or None
        project_config=accelerator_project_config,
    )

    # Logging per process
    # Use accelerator's logger for process-aware logging
    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=False)
    if accelerator.is_local_main_process:
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    logger.info(f"Running experiment: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    import json
    logger.info(f"Loaded configuration:\n{json.dumps(config, indent=2)}")

    if config.get("seed") is not None:
        set_seed(config["seed"])
        logger.info(f"Set random seed to {config['seed']}")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        # logging_dir for accelerator is already created by ProjectConfiguration
        config_save_path = os.path.join(output_dir, "config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_save_path}")

    wandb_entity = logging_config.get("entity", None)
    use_wandb = accelerator.is_main_process and report_to in ["wandb", "all"]

    if use_wandb:
        try:
            wandb.init(
                project=config.get("project_name", "vae-channel-dynamics"),
                name=run_name,
                config=config,  # Log the full config
                dir=output_dir,  # Store wandb files in output_dir/wandb
                entity=wandb_entity,
            )
            logger.info(f"Weights & Biases initialized (Entity: {wandb_entity or 'default'}).")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}. Continuing without wandb.")
            use_wandb = False
            if accelerator.log_with is None and report_to != "none":  # Fallback if W&B fails and no other logger
                accelerator.log_with = "tensorboard"
                logger.warning("Falling back to accelerator logging with tensorboard due to W&B init failure.")

    accelerator.wait_for_everyone()  # Ensure all processes are synced after setup

    model_config = config.get("model", {})
    model_torch_dtype = None
    if mixed_precision_config == "fp16":
        model_torch_dtype = torch.float16
    elif mixed_precision_config == "bf16":
        model_torch_dtype = torch.bfloat16

    vae_wrapper = SDXLVAEWrapper(
        pretrained_model_name_or_path=model_config.get("pretrained_vae_name", "stabilityai/sdxl-vae"),
        torch_dtype=model_torch_dtype
    )

    data_config = config.get("data", {})
    train_dataset = load_and_preprocess_dataset(
        dataset_name=data_config.get("dataset_name"),
        dataset_config_name=data_config.get("dataset_config_name", None),
        image_column=data_config.get("image_column", "image"),
        resolution=data_config.get("resolution", 256),
        max_samples=data_config.get("max_samples", None),
        split="train",  # Assuming "train" split for training
    )
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=data_config.get("batch_size", 4),
        num_workers=data_config.get("num_workers", 0),
        shuffle=True
    )

    training_config = config.get("training", {})
    try:
        learning_rate = float(training_config.get("learning_rate", 1e-5))
    except (ValueError, TypeError):
        learning_rate = 1e-5
        logger.error(f"Invalid learning rate format. Using default {learning_rate}.")

    optimizer = torch.optim.AdamW(
        vae_wrapper.parameters(),  # Pass VAE parameters to optimizer
        lr=learning_rate,
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        weight_decay=training_config.get("adam_weight_decay", 1e-2),
        eps=training_config.get("adam_epsilon", 1e-08),
    )

    try:
        num_samples_train = len(train_dataset) if hasattr(train_dataset, "__len__") else None
        if num_samples_train:
            num_update_steps_per_epoch = math.ceil(
                num_samples_train / data_config.get("batch_size", 4) / training_config.get(
                    "gradient_accumulation_steps", 1))
        else:  # Iterable dataset case
            num_update_steps_per_epoch = training_config.get("max_steps_per_epoch_iterable",
                                                             10000)  # Configurable fallback
            logger.warning(
                f"Dataset length unknown (likely iterable). Using max_steps_per_epoch_iterable: {num_update_steps_per_epoch}.")
    except TypeError:  # Should not happen if hasattr is used correctly
        num_update_steps_per_epoch = training_config.get("max_steps_per_epoch_iterable", 10000)
        logger.warning(
            f"Could not determine dataset length. Using max_steps_per_epoch_iterable: {num_update_steps_per_epoch}.")

    num_train_epochs = int(training_config.get("num_train_epochs", 1))
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_warmup_steps = int(training_config.get("lr_warmup_steps", 100))

    # LR Scheduler (LambdaLR example)
    def lr_lambda(current_step: int):
        if current_step < lr_warmup_steps:
            return float(current_step) / float(max(1, lr_warmup_steps))
        # Cosine decay after warmup
        # progress = float(current_step - lr_warmup_steps) / float(max(1, max_train_steps - lr_warmup_steps))
        # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # Linear decay after warmup
        decay_steps = max(1, max_train_steps - lr_warmup_steps)
        progress = float(current_step - lr_warmup_steps) / float(decay_steps)
        progress = min(1.0, progress)  # Ensure progress doesn't exceed 1.0
        return max(0.0, 1.0 - progress)  # Linear decay from 1 to 0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("Preparing components with Accelerator...")
    prepared_vae_wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae_wrapper, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Preparation complete.")

    unwrapped_model = accelerator.unwrap_model(prepared_vae_wrapper)

    dead_neuron_tracker_config = config.get("dead_neuron_tracking", {})
    dead_neuron_tracker = None  # Initialize to None
    if dead_neuron_tracker_config.get("enabled", True):  # Check if enabled in config
        dead_neuron_tracker = DeadNeuronTracker(
            target_layer_classes=target_layer_classes,
            target_layer_names_for_raw_weights=dead_neuron_tracker_config.get("target_layer_names_for_raw_weights", []),
            threshold=threshold_dn,
            mean_percentage=mean_percentage_dn,
            dead_type=dead_type_dn
        )
        logger.info("DeadNeuronTracker initialized.")
    else:
        logger.info("DeadNeuronTracker is disabled in the configuration.")

    monitor_config = config.get("tracking", {})
    monitor = ActivityMonitor(prepared_vae_wrapper, monitor_config)

    classifier_config = config.get("classification", {})
    classifier = RegionClassifier(
        model=unwrapped_model.vae if hasattr(unwrapped_model, 'vae') else unwrapped_model,  # Pass the actual VAE
        config=classifier_config
    ) if classifier_config.get("enabled", False) else None

    intervention_config = config.get("intervention", {})
    intervention_handler = None
    if intervention_config.get("enabled", False) and accelerator.is_main_process:
        try:
            intervention_handler = InterventionHandler(
                model=unwrapped_model.vae if hasattr(unwrapped_model, 'vae') else unwrapped_model,
                # Pass the actual VAE
                config=intervention_config
            )
            logger.info("InterventionHandler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize InterventionHandler: {e}", exc_info=True)
            intervention_handler = None

    logit_lens_config = config.get("logit_lens", {})
    logit_lens_analyzer = None
    if logit_lens_config.get("enabled", False) and accelerator.is_main_process:
        try:
            logit_lens_analyzer = VAELogitLens(
                model_for_lens=unwrapped_model.vae if hasattr(unwrapped_model, 'vae') else unwrapped_model,
                # Pass the actual VAE
                logit_lens_config=logit_lens_config,
                main_experiment_output_dir=output_dir
            )
            logger.info("VAELogitLens initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize VAELogitLens: {e}", exc_info=True)
            logit_lens_analyzer = None

    total_batch_size = data_config.get("batch_size", 4) * accelerator.num_processes * training_config.get(
        "gradient_accumulation_steps", 1)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_samples_train if num_samples_train else 'Unknown (Iterable)'}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {data_config.get('batch_size', 4)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_config.get('gradient_accumulation_steps', 1)}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0  # For resuming, not implemented here yet

    try:
        kl_weight = float(training_config.get("kl_weight", 1e-6))
    except (ValueError, TypeError):
        kl_weight = 1e-6
        logger.error(f"Invalid kl_weight format. Using default {kl_weight}.")

    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    log_interval = logging_config.get("log_interval", 10)
    save_interval_steps = config.get("saving", {}).get("save_interval_steps", 500)  # Changed for clarity
    checkpoint_dir_prefix = config.get("saving", {}).get("checkpoint_dir_prefix", "chkpt")

    # This is ActivityMonitor's processing interval
    activity_monitor_track_interval = monitor_config.get("track_interval", 100)

    # This interval is for DeadNeuronTracker (weights) - can be same or different
    dead_neuron_track_interval = dead_neuron_tracker_config.get("track_interval", activity_monitor_track_interval)

    logit_lens_visualization_interval = logit_lens_config.get("visualization_interval", 1000)

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )

    # Training loop
    for epoch in range(first_epoch, num_train_epochs):
        prepared_vae_wrapper.train()
        train_loss_accum = 0.0
        rec_loss_accum = 0.0
        kl_loss_accum = 0.0
        step_count_in_epoch = 0

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch.get("pixel_values")  # From custom collate_fn
            if pixel_values is None or not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4 or \
                    pixel_values.shape[0] == 0:
                logger.warning(
                    f"Step {global_step} (Epoch {epoch}, Batch {step}): Invalid batch data received (pixel_values is None or malformed), skipping.")
                continue

            with accelerator.accumulate(prepared_vae_wrapper):
                # Forward pass
                model_output = prepared_vae_wrapper(pixel_values, sample_posterior=True)
                reconstruction = model_output["reconstruction"]
                latent_dist = model_output["latent_dist"]

                # Loss calculation
                rec_loss = F.mse_loss(reconstruction.float(), pixel_values.float(), reduction="mean")
                kl_loss = latent_dist.kl().mean()
                total_loss = rec_loss + kl_weight * kl_loss

                # Gather losses for logging (especially in distributed training)
                # Effective batch size for this gathered loss calculation:
                effective_batch_size = pixel_values.shape[0] * accelerator.num_processes

                avg_loss = accelerator.gather(total_loss.detach().unsqueeze(0).expand(pixel_values.shape[0], -1)).mean()
                avg_rec_loss = accelerator.gather(
                    rec_loss.detach().unsqueeze(0).expand(pixel_values.shape[0], -1)).mean()
                avg_kl_loss = accelerator.gather(kl_loss.detach().unsqueeze(0).expand(pixel_values.shape[0], -1)).mean()

                train_loss_accum += avg_loss.item()
                rec_loss_accum += avg_rec_loss.item()
                kl_loss_accum += avg_kl_loss.item()
                step_count_in_epoch += 1

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    if max_grad_norm is not None and max_grad_norm > 0:  # Check if > 0
                        params_to_clip = [p for p in prepared_vae_wrapper.parameters() if p.requires_grad]
                        if params_to_clip: accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # More memory efficient

                    global_step += 1
                    progress_bar.update(1)

                    # --- Activity Monitoring ---
                    # monitor.step() processes buffered hook data and returns metrics for logging
                    activity_metrics_for_log = {}
                    if monitor is not None and monitor_config.get("enabled",
                                                                  False) and global_step % activity_monitor_track_interval == 0:
                        activity_metrics_for_log = monitor.step(
                            global_step)  # monitor.step now tied to its own interval

                    # --- Live Classification and Intervention ---
                    classification_output_for_intervention = {}
                    if classifier is not None and classifier_config.get("enabled",
                                                                        False) and accelerator.is_main_process:
                        # Ensure data is available from monitor (it processes based on its own track_interval)
                        if global_step % activity_monitor_track_interval == 0:  # Align with when monitor processed data
                            tracked_data_for_classifier = monitor.get_data_for_step(global_step)
                            if tracked_data_for_classifier:
                                logger.info(
                                    f"Step {global_step}: Feeding tracked data to classifier. Monitored layers: {list(tracked_data_for_classifier.keys())}")
                                classification_output_for_intervention = classifier.classify(
                                    tracked_data_for_classifier, global_step
                                )
                                if not classification_output_for_intervention:
                                    logger.info(
                                        f"Step {global_step}: Classifier returned no inactive channels to target.")
                            else:
                                logger.info(
                                    f"Step {global_step}: No tracked data available from monitor for classifier (monitor interval: {activity_monitor_track_interval}). Skipping classification.")
                        else:
                            logger.debug(
                                f"Step {global_step}: Not an ActivityMonitor processing step. Skipping classification.")

                    if intervention_handler is not None and intervention_config.get("enabled",
                                                                                    False) and accelerator.is_main_process:
                        intervention_is_due = (global_step % intervention_config.get("intervention_interval",
                                                                                     200) == 0)  # Default from example config

                        if intervention_is_due:
                            if classification_output_for_intervention:
                                logger.info(
                                    f"Step {global_step}: Intervention due. Passing {len(classification_output_for_intervention)} classified region(s) to InterventionHandler.")
                                intervention_handler.intervene(classification_output_for_intervention, global_step)
                            else:
                                logger.info(
                                    f"Step {global_step}: Intervention due, but no regions were classified for intervention by RegionClassifier.")
                    # --- End Live Classification and Intervention ---

                    # --- Logging ---
                    if global_step % log_interval == 0:
                        if step_count_in_epoch > 0:  # Avoid division by zero if log_interval < grad_accum_steps
                            current_lr = lr_scheduler.get_last_lr()[0]
                            logs = {
                                "train_loss_step": avg_loss.item(),  # Log the non-accumulated step loss
                                "reconstruction_loss_step": avg_rec_loss.item(),
                                "kl_loss_step": avg_kl_loss.item(),
                                "lr": current_lr,
                                "epoch": epoch,
                                # "step_in_epoch": step_count_in_epoch # Less useful than global_step
                            }
                            # Add accumulated averages if desired for smoother plot, but step loss is more direct
                            # logs["avg_train_loss_accum"] = train_loss_accum / step_count_in_epoch

                            if activity_metrics_for_log:  # Add metrics from monitor.step()
                                logs.update(activity_metrics_for_log)

                            accelerator.log(logs, step=global_step)  # For TensorBoard if configured
                            if use_wandb and wandb.run is not None:
                                wandb.log(logs, step=global_step)

                            progress_bar.set_postfix(
                                **{k: f"{v:.4e}" if isinstance(v, float) else v for k, v in logs.items() if
                                   k in ['train_loss_step', 'lr', 'epoch']})

                            # Reset accumulators IF log_interval aligns with full processing of accumulated batch
                            # This logic is simpler if log_interval is a multiple of gradient_accumulation_steps
                            # For now, just log step loss. Accumulators are for epoch avg.
                        else:
                            logger.debug(
                                f"Step {global_step}: log_interval hit but step_count_in_epoch is 0. Skipping log update for accumulators.")

                    # --- VAELogitLens Visualization ---
                    if logit_lens_analyzer is not None and \
                            logit_lens_config.get("enabled", False) and \
                            accelerator.is_local_main_process and \
                            global_step % logit_lens_visualization_interval == 0:
                        logger.info(f"Step {global_step}: Attempting VAELogitLens visualizations...")
                        # Ensure activation data is from the current step if monitor's interval aligns
                        # LogitLens might need data even if monitor didn't just process, if it has persistent data
                        current_step_activation_data_for_lens = monitor.get_data_for_step(global_step)
                        if not current_step_activation_data_for_lens and global_step % activity_monitor_track_interval != 0:
                            # If monitor didn't run this step, maybe LogitLens can use last available data.
                            # For now, let's assume it needs fresh data if monitor would have run.
                            logger.warning(
                                f"LogitLens: No fresh activation data from monitor for step {global_step}. Trying to get most recent if available.")
                            # Find most recent step data if monitor did not run this exact step.
                            # This might be complex; simpler to align logit_lens_interval with monitor_interval for now.

                        if current_step_activation_data_for_lens:
                            logit_lens_analyzer.run_logit_lens_with_activations(
                                global_step=global_step,
                                layers_to_analyze=logit_lens_config.get("layers_to_analyze_direct", []),  # Configurable
                                num_batch_samples_to_viz=logit_lens_config.get("num_batch_samples_to_viz", 1),
                                projection_type=logit_lens_config.get("projection_type", "mini_decoder_single_channel"),
                                activations_to_process=current_step_activation_data_for_lens,
                            )
                        else:
                            logger.warning(
                                f"LogitLens: No activation data from monitor for step {global_step} (monitor interval: {activity_monitor_track_interval}). Skipping viz.")

                    # --- Dead Neuron (Weight) Tracking ---
                    if dead_neuron_tracker is not None and dead_neuron_tracker_config.get("enabled", True) and \
                            global_step % dead_neuron_track_interval == 0:
                        logger.debug(f"Step {global_step}: Tracking dead neuron weights.")
                        dead_neuron_tracker.track_dead_neurons(
                            unwrapped_model.vae if hasattr(unwrapped_model, 'vae') else unwrapped_model)

                    # --- Saving Checkpoints ---
                    if global_step % save_interval_steps == 0:
                        if accelerator.is_main_process:
                            chkpt_save_dir = os.path.join(output_dir, f"{checkpoint_dir_prefix}-{global_step}")
                            try:
                                accelerator.save_state(chkpt_save_dir)  # Saves optimizer, scheduler, model EMA, etc.
                                logger.info(f"Saved Accelerator checkpoint state to {chkpt_save_dir}")
                                # Optionally save unwrapped model separately if needed for easier loading elsewhere
                                # unwrapped_model_to_save = accelerator.unwrap_model(prepared_vae_wrapper)
                                # unwrapped_model_to_save.vae.save_pretrained(os.path.join(chkpt_save_dir, "vae_unwrapped"))

                            except Exception as save_e:
                                logger.error(f"Error saving checkpoint state: {save_e}")

            if global_step >= max_train_steps:
                break
                # End of epoch
        avg_epoch_loss = train_loss_accum / step_count_in_epoch if step_count_in_epoch > 0 else 0
        avg_epoch_rec_loss = rec_loss_accum / step_count_in_epoch if step_count_in_epoch > 0 else 0
        avg_epoch_kl_loss = kl_loss_accum / step_count_in_epoch if step_count_in_epoch > 0 else 0

        logger.info(
            f"Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4e}, Avg Rec Loss: {avg_epoch_rec_loss:.4e}, Avg KL Loss: {avg_epoch_kl_loss:.4e}")
        epoch_logs = {
            "train_loss_epoch": avg_epoch_loss,
            "reconstruction_loss_epoch": avg_epoch_rec_loss,
            "kl_loss_epoch": avg_epoch_kl_loss,
            "epoch_num": epoch
        }
        accelerator.log(epoch_logs, step=global_step)  # Log epoch averages at the end of epoch's global_step
        if use_wandb and wandb.run is not None:
            wandb.log(epoch_logs, step=global_step)

        if global_step >= max_train_steps:
            logger.info("Reached max_train_steps. Exiting training.")
            break

    # End of training
    accelerator.wait_for_everyone()
    logger.info("Training finished.")

    if use_wandb and wandb.run is not None:
        logger.info("Waiting a few seconds before finishing W&B run...")
        time.sleep(5)  # Short delay for W&B to sync

    if accelerator.is_main_process:
        final_model_save_path = os.path.join(output_dir, "final_model_state")
        logger.info(f"Saving final Accelerator training state to {final_model_save_path}")
        try:
            accelerator.save_state(final_model_save_path)

            final_vae_save_path = os.path.join(output_dir, "final_vae_unwrapped")
            logger.info(f"Saving final unwrapped VAE model to {final_vae_save_path}")
            # Ensure we save the core VAE model correctly
            core_vae_model = unwrapped_model.vae if hasattr(unwrapped_model, 'vae') else unwrapped_model
            core_vae_model.save_pretrained(final_vae_save_path)
            logger.info(f"Final unwrapped VAE saved to {final_vae_save_path}")

        except Exception as final_save_e:
            logger.error(f"Error saving final model/state: {final_save_e}")

        if monitor is not None and monitor_config.get("enabled", False):
            logger.info("Exporting tracked activation statistics to CSV...")
            activation_stats_records = monitor.export_all_processed_data_to_records()
            if activation_stats_records:
                activation_df = pd.DataFrame(activation_stats_records)
                activation_csv_path = os.path.join(output_dir, "tracked_activation_stats.csv")
                try:
                    activation_df.to_csv(activation_csv_path, index=False)
                    logger.info(f"Saved tracked activation stats to {activation_csv_path}")
                    if use_wandb and wandb.run is not None and wandb.run.id:
                        artifact_name = f"{run_name}_activation_stats"
                        # Sanitize artifact name if needed, W&B has restrictions
                        artifact_name = "".join(
                            c if c.isalnum() or c in ('-', '_', '.') else '_' for c in artifact_name)

                        artifact = wandb.Artifact(artifact_name, type='dataset')
                        artifact.add_file(activation_csv_path)
                        wandb.log_artifact(artifact)
                        logger.info(f"Logged {activation_csv_path} to W&B artifacts as '{artifact_name}'.")
                except Exception as csv_e:
                    logger.error(f"Failed to save activation stats CSV or log to W&B: {csv_e}")
            else:
                logger.info("No activation stats records to save to CSV from ActivityMonitor.")

        if dead_neuron_tracker is not None and dead_neuron_tracker_config.get("enabled", True):
            logger.info("Saving dead neuron (weight) plots...")
            plotter = DeadNeuronPlotter(
                threshold=threshold_dn,  # Use the correct threshold
                output_dir=output_dir,
                track_interval=dead_neuron_track_interval  # Use the interval for this tracker
            )
            plotter.plot_all(
                percent_history=dead_neuron_tracker.percent_history,
                weights_history=dead_neuron_tracker.weights_history
            )
            logger.info(f"Dead neuron plots saved to {output_dir}")

        if use_wandb and wandb.run is not None:
            try:
                wandb.finish()
                logger.info("W&B run finished.")
            except Exception as wb_e:
                logger.error(f"Error finishing W&B: {wb_e}")

    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure a logger is available for this top-level exception
        # If setup_logging hasn't run or failed, this might go to default stderr
        logging.getLogger(__name__).error(f"An unhandled exception occurred in main: {e}", exc_info=True)
        sys.exit(1)