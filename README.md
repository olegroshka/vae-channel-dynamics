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
    git clone <repo-url>
    cd vae-channel-dynamics
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Login to Weights & Biases (if using):**
    ```bash
    wandb login
    ```
5.  **Set Hugging Face Token (if needed for private models/datasets):**
    ```bash
    export HF_TOKEN='your_hf_token_here'
    # Or place it in your environment variables
    ```

## Usage

Run training using the main script, specifying a configuration file:

```bash
accelerate
