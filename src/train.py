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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import yaml
import pandas as pd

# Local imports
from utils.config_utils import load_config
from utils.logging_utils import setup_logging
from utils.plotting_utils import DeadNeuronPlotter, ActivityPlotter
from utils.plotting_utils import plot_dead_vs_nudge
from data_utils import load_and_preprocess_dataset, create_dataloader
from models.sdxl_vae_wrapper import SDXLVAEWrapper
from tracking.monitor import ActivityMonitor
from tracking.deadneuron import DeadNeuronTracker
from classification.classifier import RegionClassifier
from intervention.nudger import InterventionHandler
from analysis.logit_lens import VAELogitLens

setup_logging()
logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore", category=UserWarning)

target_layer_classes = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear, torch.nn.GroupNorm)


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


def run_validation(
        accelerator: Accelerator,
        model: torch.nn.Module,
        val_dataloader: DataLoader,
        kl_weight: float,
        global_step: int,
        use_wandb_val: bool
):
    logger.info(f"--- Running Validation for Global Step: {global_step} ---")
    model.eval()
    total_val_loss_sum, total_val_rec_loss_sum, total_val_kl_loss_sum = 0.0, 0.0, 0.0
    num_val_samples_processed = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False,
                          disable=not accelerator.is_local_main_process):
            pixel_values = batch.get("pixel_values")
            if pixel_values is None or pixel_values.ndim != 4 or pixel_values.shape[0] == 0:
                logger.warning(f"Validation: Invalid batch data, skipping.")
                continue
            batch_size = pixel_values.shape[0]
            input_dtype = next(model.parameters()).dtype
            pixel_values = pixel_values.to(accelerator.device, dtype=input_dtype)
            model_output = model(pixel_values, sample_posterior=False)
            reconstruction, latent_dist = model_output["reconstruction"], model_output["latent_dist"]
            rec_loss = F.mse_loss(reconstruction.float(), pixel_values.float(), reduction="sum")
            kl_div = latent_dist.kl().sum()
            gathered_rec_loss = accelerator.gather(rec_loss.detach())
            gathered_kl_div = accelerator.gather(kl_div.detach())
            total_val_rec_loss_sum += gathered_rec_loss.sum().item()
            total_val_kl_loss_sum += gathered_kl_div.sum().item()
            num_val_samples_processed += batch_size * accelerator.num_processes
    avg_val_rec_loss = total_val_rec_loss_sum / num_val_samples_processed if num_val_samples_processed > 0 else 0
    avg_val_kl_loss = total_val_kl_loss_sum / num_val_samples_processed if num_val_samples_processed > 0 else 0
    avg_val_loss = avg_val_rec_loss + kl_weight * avg_val_kl_loss
    logger.info(f"--- Validation Complete for Global Step: {global_step} ---")
    logger.info(
        f"  Avg Validation Loss (Total): {avg_val_loss:.4e}, Rec: {avg_val_rec_loss:.4e}, KL: {avg_val_kl_loss:.4e}")
    logger.info(f"  Validated on {num_val_samples_processed} samples.")
    val_metrics = {"validation/avg_total_loss": avg_val_loss, "validation/avg_reconstruction_loss": avg_val_rec_loss,
                   "validation/avg_kl_divergence": avg_val_kl_loss}
    if accelerator.is_main_process:
        accelerator.log(val_metrics, step=global_step)
        if use_wandb_val and wandb.run is not None: wandb.log(val_metrics, step=global_step)
    model.train()
    return val_metrics


def main():
    args = parse_args()
    config = load_config(args.config_path)
    run_name = config.get("run_name", "vae_run")
    threshold_dn = float(config.get("threshold", 1e-8))
    mean_percentage_dn = float(config.get("mean_percentage", .01))
    dead_type_dn = config.get("dead_type", "threshold")
    output_dir = os.path.join(config.get("output_dir", "./results"), run_name)  # Main output dir for this run
    logging_dir = os.path.join(output_dir, "logs")
    logging_config_dict = config.get("logging", {})
    report_to = logging_config_dict.get("report_to", "tensorboard")
    mixed_precision_config = config.get("training", {}).get("mixed_precision", "no")
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    log_with_accelerate = None
    if report_to in ["wandb", "all"]:
        log_with_accelerate = None
    elif report_to == "none":
        log_with_accelerate = None
    else:
        log_with_accelerate = report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision_config, log_with=log_with_accelerate, project_config=accelerator_project_config)
    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=False)
    if accelerator.is_local_main_process:
        logging.getLogger("datasets").setLevel(logging.INFO)
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger.info(f"Running experiment: {run_name}")
    import json
    logger.info(f"Loaded configuration:\n{json.dumps(config, indent=2)}")
    if config.get("seed") is not None: set_seed(config["seed"]); logger.info(f"Set seed to {config['seed']}")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)  # Ensure main run output_dir exists
        with open(os.path.join(output_dir, "config.yaml"), 'w') as f: yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved config to {os.path.join(output_dir, 'config.yaml')}")
    wandb_entity = logging_config_dict.get("entity", None)
    use_wandb = accelerator.is_main_process and report_to in ["wandb", "all"]
    if use_wandb:
        try:
            wandb.init(project=config.get("project_name", "vae_project"), name=run_name, config=config, dir=output_dir,
                       entity=wandb_entity)
            logger.info(f"W&B initialized (Entity: {wandb_entity or 'default'}).")
        except Exception as e:
            logger.error(f"W&B init failed: {e}. No W&B logging."); use_wandb = False
    accelerator.wait_for_everyone()
    model_config = config.get("model", {})
    model_torch_dtype = None
    if mixed_precision_config == "fp16":
        model_torch_dtype = torch.float16
    elif mixed_precision_config == "bf16":
        model_torch_dtype = torch.bfloat16
    vae_wrapper = SDXLVAEWrapper(
        pretrained_model_name_or_path=model_config.get("pretrained_vae_name", "stabilityai/sdxl-vae"),
        torch_dtype=model_torch_dtype)
    data_config = config.get("data", {})
    train_dataset = load_and_preprocess_dataset(
        dataset_name=data_config.get("dataset_name"), dataset_config_name=data_config.get("dataset_config_name", None),
        image_column=data_config.get("image_column", "image"), resolution=data_config.get("resolution", 256),
        max_samples=data_config.get("max_samples", None), split=data_config.get("train_split_name", "train"))
    train_dataloader = create_dataloader(
        train_dataset, batch_size=data_config.get("batch_size", 4),
        num_workers=data_config.get("num_workers", 0), shuffle=True)
    val_dataloader = None
    if data_config.get("do_validation", False):
        logger.info("Validation enabled. Loading validation dataset...")
        try:
            validation_dataset = load_and_preprocess_dataset(
                dataset_name=data_config.get("validation_dataset_name", data_config.get("dataset_name")),
                dataset_config_name=data_config.get("validation_dataset_config_name",
                                                    data_config.get("dataset_config_name", None)),
                image_column=data_config.get("image_column", "image"), resolution=data_config.get("resolution", 256),
                max_samples=data_config.get("validation_max_samples", None),
                split=data_config.get("validation_split_name", "validation"))
            val_dataloader = create_dataloader(validation_dataset, batch_size=data_config.get("validation_batch_size",
                                                                                              data_config.get(
                                                                                                  "batch_size", 4)),
                                               num_workers=data_config.get("num_workers", 0), shuffle=False)
            logger.info(
                f"Validation dataloader created. Samples: {len(validation_dataset) if hasattr(validation_dataset, '__len__') else 'Iterable'}")
        except Exception as e:
            logger.error(f"Failed to load validation data: {e}. Disabling validation."); data_config[
                "do_validation"] = False
    training_config = config.get("training", {})
    optimizer = torch.optim.AdamW(
        vae_wrapper.parameters(), lr=float(training_config.get("learning_rate", 1e-5)),
        betas=(training_config.get("adam_beta1", 0.9), training_config.get("adam_beta2", 0.999)),
        weight_decay=training_config.get("adam_weight_decay", 1e-2), eps=training_config.get("adam_epsilon", 1e-08))
    num_samples_train = len(train_dataset) if hasattr(train_dataset, "__len__") else None
    num_update_steps_per_epoch = math.ceil(
        num_samples_train / data_config.get("batch_size", 4) / training_config.get("gradient_accumulation_steps",
                                                                                   1)) if num_samples_train else training_config.get(
        "max_steps_per_epoch_iterable", 10000)
    num_train_epochs = int(training_config.get("num_train_epochs", 1))
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_warmup_steps = int(training_config.get("lr_warmup_steps", 100))

    def lr_lambda_fn(current_step: int):
        if current_step < lr_warmup_steps: return float(current_step) / float(max(1, lr_warmup_steps))
        progress = float(current_step - lr_warmup_steps) / float(max(1, max_train_steps - lr_warmup_steps))
        return max(0.0, 1.0 - min(1.0, progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)
    logger.info("Preparing components with Accelerator...")
    if val_dataloader and data_config.get("do_validation", False):
        prepared_vae_wrapper, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            vae_wrapper, optimizer, train_dataloader, val_dataloader, lr_scheduler)
        logger.info("Train and Validation DataLoaders prepared.")
    else:
        prepared_vae_wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            vae_wrapper, optimizer, train_dataloader, lr_scheduler)
        logger.info("Train DataLoader prepared. Validation Dataloader not prepared/disabled.")
    unwrapped_model = accelerator.unwrap_model(prepared_vae_wrapper)
    core_vae_model_for_components = unwrapped_model.vae if hasattr(unwrapped_model,
                                                                   'vae') and unwrapped_model.vae is not None else unwrapped_model

    dead_neuron_tracker_config = config.get("dead_neuron_tracking", {})
    dead_neuron_tracker = None
    if dead_neuron_tracker_config.get("enabled", False):
        dead_neuron_tracker = DeadNeuronTracker(
            target_layer_classes=target_layer_classes,
            target_layer_names_for_raw_weights=dead_neuron_tracker_config.get("target_layer_names_for_raw_weights", []),
            threshold=threshold_dn, mean_percentage=mean_percentage_dn, dead_type=dead_type_dn
        )
        logger.info("DeadNeuronTracker initialized.")
    else:
        logger.info("DeadNeuronTracker disabled.")

    monitor_config = config.get("tracking", {})
    monitor = ActivityMonitor(prepared_vae_wrapper, monitor_config) if monitor_config.get("enabled", False) else None
    if monitor:
        logger.info("ActivityMonitor initialized.")
    else:
        logger.info("ActivityMonitor disabled.")

    classifier_config = config.get("classification", {})
    classifier = RegionClassifier(model=core_vae_model_for_components,
                                  config=classifier_config) if classifier_config.get("enabled", False) else None
    if classifier:
        logger.info("RegionClassifier initialized.")
    else:
        logger.info("RegionClassifier disabled.")

    intervention_config = config.get("intervention", {})
    intervention_handler = InterventionHandler(model=core_vae_model_for_components,
                                               config=intervention_config) if intervention_config.get("enabled",
                                                                                                      False) and accelerator.is_main_process else None
    if intervention_handler:
        logger.info("InterventionHandler initialized for main process.")
    else:
        logger.info("InterventionHandler disabled or not main process.")

    logit_lens_config = config.get("logit_lens", {})
    logit_lens_analyzer = VAELogitLens(model_for_lens=core_vae_model_for_components,
                                       logit_lens_config=logit_lens_config,
                                       main_experiment_output_dir=output_dir) if logit_lens_config.get("enabled",
                                                                                                       False) and accelerator.is_main_process else None
    if logit_lens_analyzer:
        logger.info("VAELogitLens initialized for main process.")
    else:
        logger.info("VAELogitLens disabled or not main process.")

    logger.info("***** Running training *****")  # ... (Log training params)
    global_step = 0
    kl_weight = float(training_config.get("kl_weight", 1e-6))
    max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
    log_interval = int(logging_config_dict.get("log_interval", 10))
    save_interval_steps = int(config.get("saving", {}).get("save_interval_steps", 500))
    checkpoint_dir_prefix = config.get("saving", {}).get("checkpoint_dir_prefix", "chkpt")
    activity_monitor_track_interval = int(monitor_config.get("track_interval", 100)) if monitor else -1
    dnt_config_track_interval = int(
        dead_neuron_tracker_config.get("track_interval", 100)) if dead_neuron_tracker else -1
    logit_lens_visualization_interval = int(
        logit_lens_config.get("visualization_interval", 1000)) if logit_lens_analyzer else -1
    validation_epochs_interval = int(training_config.get("validation_epochs", 0))
    validation_steps_interval = int(training_config.get("validation_steps", 0))

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Training Steps")

    for epoch in range(num_train_epochs):
        prepared_vae_wrapper.train()
        train_loss_accum, rec_loss_accum, kl_loss_accum = 0.0, 0.0, 0.0
        step_count_in_epoch = 0
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch.get("pixel_values")
            if pixel_values is None or pixel_values.ndim != 4 or pixel_values.shape[0] == 0: continue
            with accelerator.accumulate(prepared_vae_wrapper):
                model_output = prepared_vae_wrapper(pixel_values, sample_posterior=True)
                reconstruction, latent_dist = model_output["reconstruction"], model_output["latent_dist"]
                current_rec_loss = F.mse_loss(reconstruction.float(), pixel_values.float(), reduction="mean")
                current_kl_loss = latent_dist.kl().mean()
                current_total_loss = current_rec_loss + kl_weight * current_kl_loss
                avg_step_loss = accelerator.gather(current_total_loss.detach()).mean()
                avg_step_rec_loss = accelerator.gather(current_rec_loss.detach()).mean()
                avg_step_kl_loss = accelerator.gather(current_kl_loss.detach()).mean()
                train_loss_accum += avg_step_loss.item();
                rec_loss_accum += avg_step_rec_loss.item();
                kl_loss_accum += avg_step_kl_loss.item()
                step_count_in_epoch += 1
                accelerator.backward(current_total_loss)
                if accelerator.sync_gradients:
                    if max_grad_norm > 0: accelerator.clip_grad_norm_(prepared_vae_wrapper.parameters(), max_grad_norm)
                    optimizer.step();
                    lr_scheduler.step();
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1;
                    progress_bar.update(1)
                    activity_metrics_for_log = {}
                    if monitor and global_step % activity_monitor_track_interval == 0:
                        activity_metrics_for_log = monitor.step(global_step)
                    classification_output_for_intervention = {}
                    if classifier and accelerator.is_main_process and global_step % activity_monitor_track_interval == 0:
                        tracked_data = monitor.get_data_for_step(global_step) if monitor else {}
                        if tracked_data: classification_output_for_intervention = classifier.classify(tracked_data, global_step)
                        if not classification_output_for_intervention: logger.info(f"Step {global_step}: Classifier found no inactive channels.")
                    if intervention_handler and accelerator.is_main_process and global_step % intervention_config.get(
                            "intervention_interval", 200) == 0:
                        if classification_output_for_intervention:
                            logger.info(f"Step {global_step}: Applying intervention...")
                            intervention_handler.intervene(classification_output_for_intervention, global_step)
                            inactive_total = sum(
                                len(v["inactive_channel_indices"]) for v in classification_output_for_intervention.values()
                            )
                            wandb.log({"inactive_channels": inactive_total}, step=global_step)

                            scales_nudged = intervention_handler.num_nudges_applied
                            wandb.log({"nudged_scales": scales_nudged}, step=global_step)
                            with open(os.path.join(output_dir, "intervention_history.csv"), "a") as fh:
                                fh.write(f"{global_step},{inactive_total},{scales_nudged}\n")
                        else:
                            logger.info(f"Step {global_step}: Intervention due, but no regions classified.")
                    if global_step % log_interval == 0 and accelerator.is_main_process:
                        logs = {"train_loss_step": avg_step_loss.item(), "lr": lr_scheduler.get_last_lr()[0],
                                "epoch_current": epoch, **activity_metrics_for_log}
                        accelerator.log(logs, step=global_step)
                        if use_wandb: wandb.log(logs, step=global_step)
                        progress_bar.set_postfix(
                            **{k: f"{v:.3e}" if isinstance(v, float) else v for k, v in logs.items() if
                               k in ['train_loss_step', 'lr']})
                    if logit_lens_analyzer and accelerator.is_local_main_process and global_step % logit_lens_visualization_interval == 0:
                        current_step_data = monitor.get_data_for_step(global_step) if monitor else {}
                        if current_step_data:
                            logit_lens_analyzer.run_logit_lens_with_activations(
                                global_step=global_step,
                                activations_to_process=current_step_data,
                                # Pass relevant parts of logit_lens_config directly
                                layers_to_analyze=logit_lens_config.get("layers_to_analyze_direct",
                                                                        logit_lens_config.get("target_tracked_metrics",
                                                                                              [])),
                                num_batch_samples_to_viz=logit_lens_config.get("num_batch_samples_to_viz", 1),
                                projection_type=logit_lens_config.get("projection_type", "mini_decoder_single_channel")
                            )
                        else:
                            logger.warning(f"LogitLens: No activation data for step {global_step}.")

                    if dead_neuron_tracker and global_step % dnt_config_track_interval == 0:  # Use dnt_config_track_interval
                        dead_neuron_tracker.track_dead_neurons(core_vae_model_for_components, global_step)

                    if global_step % save_interval_steps == 0 and accelerator.is_main_process:
                        # Periodic checkpoints are saved under output_dir/chkpt-X
                        chkpt_save_dir = os.path.join(output_dir, f"{checkpoint_dir_prefix}-{global_step}")
                        accelerator.save_state(chkpt_save_dir)
                        logger.info(f"Saved periodic Accelerator state to {chkpt_save_dir}")
                        # Optionally save unwrapped VAE within this periodic checkpoint too
                        # core_vae_model_for_components.save_pretrained(os.path.join(chkpt_save_dir, "vae"))

                    if val_dataloader and data_config.get("do_validation",
                                                          False) and validation_steps_interval > 0 and global_step % validation_steps_interval == 0:
                        run_validation(accelerator, prepared_vae_wrapper, val_dataloader, kl_weight, global_step,
                                       use_wandb)
            if global_step >= max_train_steps: break
        if accelerator.is_main_process:
            epoch_summary_logs = {
                "train/epoch_avg_loss": train_loss_accum / step_count_in_epoch if step_count_in_epoch else float('nan'),
                "train/epoch_avg_rec_loss": rec_loss_accum / step_count_in_epoch if step_count_in_epoch else float(
                    'nan'),
                "train/epoch_avg_kl_loss": kl_loss_accum / step_count_in_epoch if step_count_in_epoch else float('nan'),
                "epoch_completed": epoch}
            accelerator.log(epoch_summary_logs, step=global_step)
            if use_wandb: wandb.log(epoch_summary_logs, step=global_step)
        logger.info(
            f"Epoch {epoch} completed. Avg Train Loss: {train_loss_accum / step_count_in_epoch if step_count_in_epoch else float('nan'):.4e}")
        if val_dataloader and data_config.get("do_validation", False) and validation_epochs_interval > 0 and (
                epoch + 1) % validation_epochs_interval == 0 and (
                validation_steps_interval is None or validation_steps_interval <= 0):
            run_validation(accelerator, prepared_vae_wrapper, val_dataloader, kl_weight, global_step, use_wandb)
        if global_step >= max_train_steps: logger.info("Reached max_train_steps."); break
    accelerator.wait_for_everyone();
    logger.info("Training finished.")
    if use_wandb and accelerator.is_main_process: time.sleep(5)

    # --- Corrected Final Model Saving ---
    if accelerator.is_main_process:
        # This is the base directory for the *final model checkpoint* that evaluate.py expects
        final_model_output_base_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_output_base_dir, exist_ok=True)

        # 1. Save Accelerator state (model.safetensors, optimizer.bin, etc.)
        #    directly into final_model_output_base_dir
        logger.info(f"Saving final Accelerator training state to {final_model_output_base_dir}")
        try:
            accelerator.save_state(final_model_output_base_dir)
            logger.info(
                f"Full training state (including wrapped model, optimizer, scheduler) saved in {final_model_output_base_dir}")
        except Exception as e:
            logger.error(f"Error saving final Accelerator state: {e}")

        # 2. Save the unwrapped VAE model into a 'vae' subdirectory
        #    This creates the structure evaluate.py expects: final_model_output_base_dir/vae/
        vae_save_path = os.path.join(final_model_output_base_dir, "vae")
        logger.info(f"Saving final unwrapped VAE model to {vae_save_path}")
        try:
            core_vae_model_for_components.save_pretrained(vae_save_path)
            logger.info(f"Final unwrapped VAE (AutoencoderKL) saved to {vae_save_path}")
        except Exception as e:
            logger.error(f"Error saving final unwrapped VAE: {e}")
        # --- End Corrected Final Model Saving ---

        activity_csv_path = None
        if monitor and monitor_config.get("enabled", False):
            activation_records = monitor.export_all_processed_data_to_records()
            if activation_records:
                df_act = pd.DataFrame(activation_records)
                # Save CSV in the main run directory (output_dir), not inside "final_model"
                activity_csv_path = os.path.join(output_dir, "tracked_activation_stats.csv")
                df_act.to_csv(activity_csv_path, index=False)
                logger.info(f"Saved activation stats to {activity_csv_path}")
                if use_wandb and wandb.run and wandb.run.id:
                    art_name = "".join(
                        c if c.isalnum() or c in ('-', '_', '.') else '_' for c in f"{run_name}_activations")
                    art = wandb.Artifact(art_name, type='dataset');
                    art.add_file(activity_csv_path);
                    wandb.log_artifact(art)

        if dead_neuron_tracker and dead_neuron_tracker_config.get("enabled", False):
            logger.info("Saving dead neuron (weight) plots...")
            # Plots will go into output_dir (e.g. ./results/your_run_name/)
            dnt_plotter = DeadNeuronPlotter(threshold=threshold_dn, output_dir=output_dir)
            dnt_plotter.plot_all(percent_history=dead_neuron_tracker.percent_history,
                                 weights_history=dead_neuron_tracker.weights_history)

        if activity_csv_path and os.path.exists(activity_csv_path):
            # Activity plots will go into output_dir/activity_plots
            activity_plot_dir = os.path.join(output_dir, "activity_plots")
            act_plotter = ActivityPlotter(output_dir=activity_plot_dir)
            act_plotter.plot_activation_stats_evolution(
                csv_path=activity_csv_path,
                target_metric_substring="mean_abs_activation_per_channel",
                target_metric_type="per_channel_overall_mean"
            )
            # Optionally add call for plotting standard deviation here if implemented
            # act_plotter.plot_activation_std_dev_evolution(csv_path=activity_csv_path, ...)
            logger.info(f"Activity evolution plots saved to {activity_plot_dir}")
        elif monitor and monitor_config.get("enabled", False):
            logger.warning("ActivityMonitor enabled, but CSV not found for plotting activity evolution.")

        if intervention_handler and intervention_handler.num_nudges_applied > 0:
            plot_dead_vs_nudge(
                csv_path=os.path.join(output_dir, "intervention_history.csv"),
                out_png=os.path.join(output_dir, "dead_vs_nudge.png"),
                nudge_factor=intervention_handler.nudge_factor
            )

        if use_wandb and wandb.run: wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"Unhandled exception in main: {e}", exc_info=True); sys.exit(1)
