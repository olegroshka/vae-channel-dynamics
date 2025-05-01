# VAE Channel Dynamics

This project investigates the phenomenon of emergent channel inactivity ("dead regions") in the SDXL VAE model. The goal is to track the appearance of these regions during training, classify them, and potentially intervene ("nudge" inactive channels) to understand their impact on model performance and behavior.

## Overview

The core components include:
- Loading and preprocessing standard image datasets.
- Fine-tuning or training the SDXL VAE model using Hugging Face `diffusers` and `accelerate`.
- Modules for tracking channel activity (activations, normalization parameters).
- Modules for classifying inactive regions based on observed dynamics.
- Modules for applying targeted interventions to potentially reactivate channels.
- Experiment configuration management using YAML files.
- Logging using standard Python logging and Weights & Biases (wandb).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:olegroshka/vae-channel-dynamics.git  
    cd vae-channel-dynamics
    ```
2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda create --name vae-dyn python=3.11 -y
    conda activate vae-dyn
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Accelerate:**
    Run the configuration wizard. It's recommended to answer 'NO' to DeepSpeed, torch dynamo, and NUMA efficiency for initial setup unless you specifically need them. Choose your hardware setup (CPU, single GPU, multi-GPU) and mixed precision preference (start with 'no' for simplicity).
    ```bash
    accelerate config
    ```
5.  **Login to Weights & Biases (Optional):**
    If you want to use wandb for logging (as configured in `base_config.yaml`), log in:
    ```bash
    wandb login
    ```
    If you prefer not to use wandb, change `report_to: "wandb"` to `report_to: "tensorboard"` in your experiment configuration files (e.g., `configs/experiment_cifar10_test.yaml`).

6.  **Set Hugging Face Token (Optional):**
    Needed if using private models or datasets from the Hugging Face Hub.
    ```bash
    export HF_TOKEN='your_hf_token_here'
    # Or add it to your shell profile (e.g., ~/.bashrc, ~/.zshrc)
    ```

## Usage

Experiments are run using the `accelerate launch` command with the main training script (`src/train.py`) and a specific configuration file from the `configs/` directory. Evaluation is run using `python src/evaluate.py`.

### Running a Test Train-Eval Loop (CIFAR-10 Example)

This example demonstrates how to run a quick training loop on a small subset of CIFAR-10 and then evaluate the resulting model to ensure the pipeline works.

1.  **Run Training:**
    Use the pre-defined test configuration `experiment_cifar10_test.yaml`. This config uses 100 samples of CIFAR-10 for 1 epoch.
    ```bash
    accelerate launch src/train.py --config_path configs/experiment_cifar10_test.yaml
    ```
    * Training logs and checkpoints will be saved under `./results/sdxl_vae_cifar10_test_run/`.
    * The final trained model (specifically the VAE component) will be saved in `./results/sdxl_vae_cifar10_test_run/final_model/vae/`.

2.  **Run Evaluation:**
    After training completes, run the evaluation script. Point it to the same configuration file (to load dataset info) and the path to the saved model checkpoint directory (`final_model`).
    ```bash
    python src/evaluate.py \
        --config_path configs/experiment_cifar10_test.yaml \
        --checkpoint_path ./results/sdxl_vae_cifar10_test_run/final_model \
        --eval_split test \
        --num_samples_to_save 10 \
        --batch_size 8
    ```
    * This will load the model saved in the previous step.
    * It will calculate metrics (MSE, KL) on the CIFAR-10 `test` split.
    * Reconstructed image samples and evaluation metrics will be saved in `./results/sdxl_vae_cifar10_test_run/final_model/eval_results_test/`.

### Running Other Experiments

To run different experiments (e.g., longer training, different datasets, enabling tracking/intervention):
1.  Create a new YAML configuration file in the `configs/` directory (you can copy and modify an existing one like `experiment_baseline.yaml` or `experiment_cifar10_test.yaml`).
2.  Adjust the parameters within the new config file (e.g., `dataset_name`, `max_samples`, `num_train_epochs`, `tracking.enabled`, `intervention.strategy`, etc.).
3.  Run training using `accelerate launch src/train.py --config_path configs/your_new_config_file.yaml`.
4.  Run evaluation using `python src/evaluate.py --config_path configs/your_new_config_file.yaml --checkpoint_path ./results/your_run_name/final_model`.

