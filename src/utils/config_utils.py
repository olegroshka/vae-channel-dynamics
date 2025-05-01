# src/utils/config_utils.py
import yaml
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file. Handles potential 'defaults' key for inheritance.

    Args:
        config_path: Path to the main YAML configuration file.

    Returns:
        A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file or any default config file doesn't exist.
        yaml.YAMLError: If there's an error parsing the YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    final_config = {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            logger.warning(f"Config file is empty: {config_path}")
            config = {}

        # Handle defaults for inheritance (simple single-level inheritance)
        if 'defaults' in config and isinstance(config['defaults'], list):
            base_config_name = config['defaults'][0] # Assuming first default is the base
            # Construct path relative to the main config file's directory
            base_config_path = os.path.join(os.path.dirname(config_path), f"{base_config_name}.yaml")
            logger.info(f"Loading base configuration from: {base_config_path}")
            if not os.path.exists(base_config_path):
                raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

            with open(base_config_path, 'r') as bf:
                base_config = yaml.safe_load(bf)
                if base_config:
                    final_config.update(base_config) # Load base first
            # Remove defaults key after processing
            del config['defaults']

        # Merge/override with specific config
        # Simple merge - nested dicts are replaced, not merged deeply
        # For deep merging, consider libraries like OmegaConf or custom logic
        final_config.update(config)

        logger.info(f"Successfully loaded configuration from {config_path}")
        return final_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise

if __name__ == '__main__':
    # Example Usage: Create dummy config files first
    dummy_base_path = os.path.join("../..", "..", "configs", "dummy_base.yaml")
    dummy_exp_path = os.path.join("../..", "..", "configs", "dummy_exp.yaml")

    os.makedirs(os.path.dirname(dummy_base_path), exist_ok=True)

    base_content = """
project_name: "dummy_project"
data:
  dataset_name: "base_dataset"
  resolution: 128
training:
  learning_rate: 1e-4
"""
    exp_content = """
defaults:
  - dummy_base # Refers to dummy_base.yaml in the same dir

run_name: "dummy_experiment_run"
data:
  resolution: 256 # Override base
training:
  num_train_epochs: 5 # Add new key
"""
    with open(dummy_base_path, 'w') as f:
        f.write(base_content)
    with open(dummy_exp_path, 'w') as f:
        f.write(exp_content)

    print(f"Created dummy configs at {os.path.abspath(os.path.dirname(dummy_base_path))}")

    try:
        loaded_config = load_config(dummy_exp_path)
        print("\nLoaded Config:")
        import json
        print(json.dumps(loaded_config, indent=2))

        # Test non-existent file
        # load_config("non_existent_config.yaml")
    except Exception as e:
        print(f"\nError during example usage: {e}")
    finally:
        # Clean up dummy files
        # os.remove(dummy_base_path)
        # os.remove(dummy_exp_path)
        print("\nCleaned up dummy files (commented out).")
        pass


