# src/train.py
import os
import sys
import argparse
import logging
import math
import warnings
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import wandb
import yaml
import pandas as pd

# Local imports
from utils.config_utils import load_config
from utils.logging_utils import setup_logging
from utils.plotting_utils import DeadNeuronPlotter
from data_utils import load_and_preprocess_dataset, create_dataloader
from models.sdxl_vae_wrapper import SDXLVAEWrapper
from tracking.monitor import ActivityMonitor
from tracking.deadneuron import DeadNeuronTracker
from classification.classifier import RegionClassifier
from intervention.nudger import InterventionHandler  # Ensure this is the updated nudger
from analysis.logit_lens import VAELogitLens

setup_logging()
logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore", category=UserWarning)

target_layer_classes = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear, torch.nn.GroupNorm)
target_layer_names_for_dead_neuron_perc = ["encoder.down_blocks.1.resnets.0.conv_shortcut.weight"]


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
        threshold = float(config.get("threshold", 1e-8))
    except (ValueError, TypeError):
        logger.error(f"Invalid threshold format: {config.get('threshold')}. Using default 1e-8.")
        threshold = 1e-8

    try:
        mean_percentage = float(config.get("mean_percentage", .01))
    except (ValueError, TypeError):
        logger.error(f"Invalid threshold format: {config.get('mean_percentage')}. Using default .01")
        mean_percentage = .01

    dead_type = config.get("dead_type", "threshold")

    output_dir = os.path.join(config.get("output_dir", "./results"), run_name)
    logging_dir = os.path.join(output_dir, "logs")
    logging_config = config.get("logging", {})
    report_to = logging_config.get("report_to", "tensorboard")
    mixed_precision_config = config.get("training", {}).get("mixed_precision", "no")

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    log_with_accelerate = None if report_to in ["wandb", "all"] else report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision_config,
        log_with=log_with_accelerate,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
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
        os.makedirs(logging_dir, exist_ok=True)
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
                config=config,
                dir=output_dir,
                entity=wandb_entity,
            )
            logger.info(f"Weights & Biases initialized (Entity: {wandb_entity or 'default'}).")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}. Continuing without wandb.")
            use_wandb = False
            if accelerator.log_with is None:
                accelerator.log_with = "tensorboard"
                logger.warning("Falling back to accelerator logging with tensorboard.")
    accelerator.wait_for_everyone()

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
        split="train",
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
        vae_wrapper.parameters(),
        lr=learning_rate,
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        weight_decay=training_config.get("adam_weight_decay", 1e-2),
        eps=training_config.get("adam_epsilon", 1e-08),
    )

    try:
        num_samples = len(train_dataset) if hasattr(train_dataset, "__len__") else None
        if num_samples:
            num_update_steps_per_epoch = math.ceil(
                num_samples / data_config.get("batch_size", 4) / training_config.get("gradient_accumulation_steps", 1))
        else:
            num_update_steps_per_epoch = 10000
            logger.warning(
                f"Dataset length unknown. Estimating num_update_steps_per_epoch as {num_update_steps_per_epoch}.")
    except TypeError:
        num_update_steps_per_epoch = 10000
        logger.warning(
            f"Could not determine dataset length. Estimating num_update_steps_per_epoch as {num_update_steps_per_epoch}.")

    num_train_epochs = int(training_config.get("num_train_epochs", 1))
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_warmup_steps = int(training_config.get("lr_warmup_steps", 100))

    def lr_lambda(current_step: int):
        if current_step < lr_warmup_steps:
            return float(current_step) / float(max(1, lr_warmup_steps))
        decay_steps = max(1, max_train_steps - lr_warmup_steps)
        progress = float(current_step - lr_warmup_steps) / float(decay_steps)
        progress = min(1.0, progress)
        return max(0.0, 1.0 - progress)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("Preparing components with Accelerator...")
    prepared_vae_wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae_wrapper, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Preparation complete.")

    # Initialize components that might need the model
    unwrapped_model = accelerator.unwrap_model(prepared_vae_wrapper)

    dead_neuron_tracker = DeadNeuronTracker(
        target_layer_classes,
        target_layer_names_for_dead_neuron_perc,
        threshold,
        mean_percentage,
        dead_type
    )

    monitor = ActivityMonitor(prepared_vae_wrapper, config.get("tracking", {}))
    classifier_config = config.get("classification", {})
    classifier = RegionClassifier(classifier_config)  # Pass its own config section

    intervention_config = config.get("intervention", {})
    intervention_handler = None
    if intervention_config.get("enabled", False) and accelerator.is_main_process:
        try:
            intervention_handler = InterventionHandler(
                model=unwrapped_model,  # Pass unwrapped model
                config=intervention_config
            )
            logger.info("InterventionHandler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize InterventionHandler: {e}", exc_info=True)
            intervention_handler = None  # Ensure it's None if initialization fails

    logit_lens_config = config.get("logit_lens", {})
    logit_lens_analyzer = None
    if logit_lens_config.get("enabled", False) and accelerator.is_main_process:
        try:
            logit_lens_analyzer = VAELogitLens(
                model_for_lens=unwrapped_model,
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
    # ... (logging of training parameters)

    global_step = 0
    first_epoch = 0
    try:
        kl_weight = float(training_config.get("kl_weight", 1e-6))
    except (ValueError, TypeError):
        kl_weight = 1e-6
        logger.error(f"Invalid kl_weight format. Using default {kl_weight}.")

    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    log_interval = logging_config.get("log_interval", 10)
    save_interval = config.get("saving", {}).get("save_interval", 500)
    checkpoint_dir_prefix = config.get("saving", {}).get("checkpoint_dir_prefix", "chkpt")

    track_interval = config.get("tracking", {}).get("track_interval", 100)
    logit_lens_visualization_interval = logit_lens_config.get("visualization_interval", 1000)

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )

    for epoch in range(first_epoch, num_train_epochs):
        prepared_vae_wrapper.train()
        train_loss_accum = 0.0
        rec_loss_accum = 0.0
        kl_loss_accum = 0.0
        step_count_in_epoch = 0

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch.get("pixel_values")
            if pixel_values is None or not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4 or \
                    pixel_values.shape[0] == 0:
                logger.warning(f"Step {global_step}: Invalid batch data, skipping.")
                continue

            with accelerator.accumulate(prepared_vae_wrapper):
                model_output = prepared_vae_wrapper(pixel_values, sample_posterior=True)
                reconstruction = model_output["reconstruction"]
                latent_dist = model_output["latent_dist"]

                rec_loss = F.mse_loss(reconstruction.float(), pixel_values.float(), reduction="mean")
                kl_loss = latent_dist.kl().mean()
                total_loss = rec_loss + kl_weight * kl_loss

                avg_loss = accelerator.gather(total_loss.detach().repeat(pixel_values.shape[0])).mean()
                avg_rec_loss = accelerator.gather(rec_loss.detach().repeat(pixel_values.shape[0])).mean()
                avg_kl_loss = accelerator.gather(kl_loss.detach().repeat(pixel_values.shape[0])).mean()

                train_loss_accum += avg_loss.item()
                rec_loss_accum += avg_rec_loss.item()
                kl_loss_accum += avg_kl_loss.item()
                step_count_in_epoch += 1

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    if max_grad_norm is not None:
                        params_to_clip = [p for p in prepared_vae_wrapper.parameters() if p.requires_grad]
                        if params_to_clip: accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    activity_metrics_for_log = monitor.step(global_step)

                    # --- Classification and Intervention ---
                    # (Currently, classifier is a placeholder, so intervention uses dummy data if enabled)
                    classification_output_for_intervention = {}  # Default to empty
                    if classifier_config.get("enabled", False) and accelerator.is_main_process:
                        # When classifier is functional, it would use data from monitor:
                        # tracked_data_for_classifier = monitor.get_data_for_step(global_step)
                        # classification_output_for_intervention = classifier.classify(tracked_data_for_classifier or {}, global_step)

                        # --- FOR TESTING NUDGER (Deliverable 3.1b) ---
                        # Manually create dummy classification_results if intervention is enabled
                        # This bypasses the placeholder classifier for now.
                        if intervention_handler is not None and intervention_config.get("enabled", False):
                            # Define a plausible GroupNorm scale parameter name from SDXL VAE
                            # IMPORTANT: User must verify this name against their model structure.
                            # This example targets an early encoder GroupNorm.
                            dummy_target_param_name = "vae.encoder.down_blocks.0.resnets.0.norm1.weight"

                            param_exists_for_nudge = False
                            try:
                                test_param_obj = intervention_handler.model  # This is the unwrapped model
                                for part in dummy_target_param_name.split('.'):
                                    test_param_obj = getattr(test_param_obj, part)
                                if isinstance(test_param_obj, torch.nn.Parameter):
                                    param_exists_for_nudge = True
                            except AttributeError:
                                logger.warning(
                                    f"Dummy target param for nudge '{dummy_target_param_name}' not found in model. Intervention with dummy data will be skipped.")

                            if param_exists_for_nudge:
                                classification_output_for_intervention = {
                                    "dummy_encoder_norm1": {  # Arbitrary key for the layer
                                        "param_name_scale": dummy_target_param_name,
                                        "inactive_channel_indices": [0, 1, 2]  # Nudge first 3 channels
                                    }
                                }
                                logger.debug(
                                    f"Step {global_step}: Created dummy classification results for nudging: {dummy_target_param_name}, channels [0,1,2]")
                            else:
                                classification_output_for_intervention = {}  # Ensure it's empty if param not found
                        # --- END TESTING NUDGER ---

                    if intervention_handler is not None and accelerator.is_main_process:
                        intervention_handler.intervene(classification_output_for_intervention, global_step)
                    # ------------------------------------

                    if global_step % log_interval == 0:
                        if step_count_in_epoch > 0:
                            avg_step_loss = train_loss_accum / step_count_in_epoch
                            avg_step_rec_loss = rec_loss_accum / step_count_in_epoch
                            avg_step_kl_loss = kl_loss_accum / step_count_in_epoch

                            logs = {
                                "train_loss": avg_step_loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "reconstruction_loss": avg_step_rec_loss,
                                "kl_loss": avg_step_kl_loss,
                                "epoch": epoch,
                                "step": global_step,
                            }

                            if activity_metrics_for_log:
                                logger.info(
                                    f"Step {global_step}: Merging activity metrics. Keys: {list(activity_metrics_for_log.keys())}")
                                logs.update(activity_metrics_for_log)
                            elif global_step % track_interval == 0:
                                logger.warning(
                                    f"Step {global_step}: activity_metrics_for_log is EMPTY, but monitor should have processed (track_interval: {track_interval}).")
                            else:
                                logger.debug(
                                    f"Step {global_step}: activity_metrics_for_log is EMPTY (global_step % track_interval != 0).")

                            progress_bar.set_postfix(
                                **{k: f"{v:.4e}" if isinstance(v, float) else v for k, v in logs.items()})

                            try:
                                if use_wandb and wandb.run is not None:
                                    wandb.log(logs, step=global_step)
                                elif accelerator.log_with is not None:
                                    accelerator.log(logs, step=global_step)
                            except Exception as log_e:
                                logger.error(f"Error during logging at step {global_step}: {log_e}")

                            train_loss_accum = 0.0
                            rec_loss_accum = 0.0
                            kl_loss_accum = 0.0
                            step_count_in_epoch = 0

                    if logit_lens_analyzer is not None and \
                            logit_lens_config.get("enabled", False) and \
                            global_step % logit_lens_visualization_interval == 0:
                        # ... (Logit Lens visualization call as before) ...
                        logger.info(f"Step {global_step}: Running VAELogitLens visualizations...")
                        current_step_activation_data = monitor.get_data_for_step(global_step)

                        if current_step_activation_data:
                            target_metrics_for_lens = logit_lens_config.get("target_tracked_metrics", [])
                            # ... (rest of logit lens call logic)
                            for target_metric_key in target_metrics_for_lens:
                                parts = target_metric_key.split('.')
                                if len(parts) < 2: continue
                                actual_metric_name = parts[-1]
                                layer_identifier = ".".join(parts[:-1])

                                if layer_identifier in current_step_activation_data and \
                                        actual_metric_name in current_step_activation_data[layer_identifier]:
                                    activation_tensor = current_step_activation_data[layer_identifier][
                                        actual_metric_name]
                                    if actual_metric_name == "full_activation_map" and isinstance(activation_tensor,
                                                                                                  torch.Tensor):
                                        logit_lens_analyzer.visualize_channel_activation_maps(
                                            activation_map_tensor=activation_tensor,
                                            layer_identifier=layer_identifier,
                                            global_step=global_step,
                                            # num_channels_to_viz, num_batch_samples_to_viz, colormap from config
                                        )
                                        if logit_lens_config.get("run_mini_decoder_projection", False):
                                            logit_lens_analyzer.project_with_mini_decoder(
                                                activation_map_tensor=activation_tensor,
                                                layer_identifier=layer_identifier,
                                                global_step=global_step
                                            )
                        else:
                            logger.warning(
                                f"LogitLens: No activation data from monitor for step {global_step}. Skipping viz.")

                    if global_step % save_interval == 0:
                        if accelerator.is_main_process:
                            chkpt_save_dir = os.path.join(output_dir, f"{checkpoint_dir_prefix}-{global_step}")
                            try:
                                accelerator.save_state(chkpt_save_dir)
                                logger.info(f"Saved checkpoint state to {chkpt_save_dir}")
                            except Exception as save_e:
                                logger.error(f"Error saving checkpoint state: {save_e}")

                    if global_step % track_interval == 0:
                        dead_neuron_tracker.track_dead_neurons(accelerator.unwrap_model(vae_wrapper))

            if global_step >= max_train_steps: break
        logger.info(f"Epoch {epoch} completed.")

        if accelerator.is_main_process:
            # ... (existing dead neuron weight percentage logging) ...
            pass
        if global_step >= max_train_steps:
            logger.info("Reached max_train_steps. Exiting training.")
            break

    accelerator.wait_for_everyone()
    logger.info("Training finished.")
    if use_wandb and wandb.run is not None:
        logger.info("Pausing for 10 seconds before finishing wandb run...")
        time.sleep(10)

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

        logger.info("Exporting tracked activation statistics to CSV...")
        activation_stats_records = monitor.export_all_processed_data_to_records()
        if activation_stats_records:
            activation_df = pd.DataFrame(activation_stats_records)
            activation_csv_path = os.path.join(output_dir, "tracked_activation_stats.csv")
            try:
                activation_df.to_csv(activation_csv_path, index=False)
                logger.info(f"Saved tracked activation stats to {activation_csv_path}")
                if use_wandb and wandb.run is not None:
                    if wandb.run.id:
                        artifact = wandb.Artifact(f'{run_name}_activation_stats', type='dataset')
                        artifact.add_file(activation_csv_path)
                        wandb.log_artifact(artifact)
                        logger.info(f"Logged {activation_csv_path} to W&B artifacts.")
                    else:
                        logger.warning("W&B run not active, cannot log CSV artifact.")
            except Exception as csv_e:
                logger.error(f"Failed to save activation stats CSV or log to W&B: {csv_e}")
        else:
            logger.info("No activation stats records to save to CSV.")

        logger.info("Saving plots")
        plotter = DeadNeuronPlotter(threshold=threshold, output_dir=output_dir, track_interval=track_interval)
        plotter.plot_all(
            percent_history=dead_neuron_tracker.percent_history,
            weights_history=dead_neuron_tracker.weights_history)

        if use_wandb and wandb.run is not None:
            try:
                wandb.finish()
            except Exception as wb_e:
                logger.error(f"Error finishing W&B: {wb_e}")

    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled exception occurred in main: {e}", exc_info=True)
        sys.exit(1)
