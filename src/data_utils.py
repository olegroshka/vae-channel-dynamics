# src/data_utils.py
import logging
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Image, Dataset as HFDataset
from torchvision import transforms as T
from PIL import Image as PILImage # Use alias to avoid confusion

logger = logging.getLogger(__name__)

def get_transform(resolution: int) -> T.Compose:
    """
    Creates the standard image transformation pipeline for VAE training.
    Resizes, center crops, converts to RGB, converts to tensor, and normalizes to [-1, 1].

    Args:
        resolution: The target image resolution.

    Returns:
        A torchvision Compose object representing the transformation pipeline.
    """
    return T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(resolution),
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), # Ensure RGB
        T.ToTensor(),          # Converts PIL [0, 255] to Tensor [0.0, 1.0]
        T.Normalize([0.5], [0.5]), # Maps [0.0, 1.0] to [-1.0, 1.0]
    ])

def load_and_preprocess_dataset(
    dataset_name: str,
    dataset_config_name: Optional[str] = None, # <<< Parameter definition is here
    image_column: str = "image",
    resolution: int = 256,
    max_samples: Optional[int] = None,
    split: str = "train",
    streaming: bool = False, # Set to True to stream large datasets
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """
    Loads a dataset from the Hugging Face Hub, applies preprocessing,
    and optionally selects a subset.

    Args:
        dataset_name: Name or path of the dataset on the Hugging Face Hub.
        dataset_config_name: Optional configuration name for the dataset (e.g., '320px').
        image_column: Name of the column containing PIL images.
        resolution: Target resolution for image preprocessing.
        max_samples: Maximum number of samples to use (uses full dataset if None).
        split: Dataset split to load (e.g., 'train', 'test').
        streaming: Whether to stream the dataset (useful for large datasets).
        cache_dir: Optional directory for caching downloaded datasets.

    Returns:
        The processed Hugging Face Dataset object.

    Raises:
        ValueError: If the specified image column is not found or if config name is needed but not provided.
        Exception: For errors during dataset loading.
    """
    logger.info(f"Loading dataset '{dataset_name}' (Config: {dataset_config_name}, Split: {split})...")
    try:
        # Pass dataset_config_name if provided
        dataset = load_dataset(
            dataset_name,
            name=dataset_config_name, # <<< Pass config name here
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )
        logger.info("Dataset loaded successfully.")
    except ValueError as e:
        # Catch specific ValueError related to missing config name
        if "Config name is missing" in str(e):
             logger.error(f"Dataset '{dataset_name}' requires a configuration name. Error: {e}")
             logger.error("Please specify 'dataset_config_name' in your configuration YAML file.")
        else:
             logger.error(f"ValueError loading dataset '{dataset_name}': {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise

    # Verify image column exists
    if image_column not in dataset.features:
        # Attempt common alternative if primary not found
        alt_image_column = 'img' if image_column == 'image' else 'image'
        if alt_image_column in dataset.features:
            logger.warning(f"Specified image column '{image_column}' not found. Using '{alt_image_column}' instead.")
            image_column = alt_image_column
        else:
            raise ValueError(f"Image column '{image_column}' (or alternative) not found in dataset features: {dataset.features}")

    # Select subset if max_samples is specified and not streaming
    if max_samples is not None and not streaming:
        logger.info(f"Selecting the first {max_samples} samples.")
        # Ensure dataset has __len__ before comparing
        try:
            dataset_len = len(dataset)
            if max_samples > dataset_len:
                logger.warning(f"max_samples ({max_samples}) > dataset size ({dataset_len}). Using full dataset.")
            else:
                dataset = dataset.select(range(max_samples))
        except TypeError:
             logger.warning("Could not determine dataset length for max_samples check (possibly streaming or iterable dataset). Proceeding without length check.")
             # For iterable datasets, take() is the way, but select() was intended for map-style
             # If it's truly iterable, this select() might fail later or behave unexpectedly.
             # Consider adding explicit handling for IterableDataset if needed.
             pass # Allow select to proceed, might error later if not map-style

    elif max_samples is not None and streaming:
        logger.warning("max_samples is specified, but dataset is streamed. Taking the first max_samples.")
        dataset = dataset.take(max_samples)


    # Define the transformation function
    transform = get_transform(resolution)

    def transform_images(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the transform to a batch of examples."""
        processed_images = []
        images_in = examples.get(image_column) # Use .get for safety

        if images_in is None:
            logger.warning(f"Image column '{image_column}' not found in batch keys: {list(examples.keys())}. Returning empty.")
            return {"pixel_values": []}

        # Ensure images_in is always a list for consistent processing
        if not isinstance(images_in, list):
            images_in = [images_in]

        for image in images_in:
            try:
                # Check if it's a PIL Image before transforming
                if isinstance(image, PILImage.Image):
                    processed_images.append(transform(image))
                else:
                    # Log if the type is unexpected but maybe processable?
                    # Or strictly enforce PIL Image? For now, just log and skip.
                    logger.warning(f"Skipping non-PIL Image in column '{image_column}'. Type: {type(image)}")
            except Exception as e:
                logger.error(f"Error transforming image: {e}. Skipping image.", exc_info=False) # Reduce traceback noise

        # Return processed images or empty list if none were successful/valid
        return {"pixel_values": processed_images}


    logger.info(f"Applying transforms (resolution: {resolution})...")
    # Use `with_transform` for efficiency (applies transform on the fly)
    processed_dataset = dataset.with_transform(transform_images)
    logger.info("Dataset preprocessing complete.")

    # Filtering should ideally happen *after* preparing the dataloader
    # or by handling errors gracefully in the collate_fn or training loop.
    # Filtering map-style datasets here can be slow and might remove entire examples if one image fails.
    # Filtering iterable datasets here is generally not possible.

    return processed_dataset


def create_dataloader(
    dataset: HFDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    collate_fn = None, # Optional custom collate function
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset: The preprocessed Hugging Face Dataset object.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data each epoch.
        pin_memory: Whether to use pinned memory for faster GPU transfers.
        collate_fn: Optional custom function to batch data.

    Returns:
        A PyTorch DataLoader instance.
    """
    logger.info(f"Creating DataLoader (Batch size: {batch_size}, Shuffle: {shuffle}, Workers: {num_workers})")
    # Note: If using streaming dataset, shuffle=True might behave differently or not be supported fully.
    # Consider `IterableDataset` specifics if streaming is heavily used.
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    if is_iterable and shuffle:
        logger.warning("Shuffle=True may have limited effect with IterableDataset (streaming).")

    # Use default_collate if collate_fn is None
    collate_to_use = collate_fn if collate_fn is not None else torch.utils.data.default_collate

    # Filter out None pixel_values potentially returned by transform_images if errors occurred
    # This is a basic filter; more robust error handling might be needed
    def safe_collate(batch):
        # Filter out items where pixel_values might be missing or empty after transform errors
        # Assuming 'pixel_values' is the key added by transform_images
        filtered_batch = [item for item in batch if item.get("pixel_values") is not None and len(item["pixel_values"]) > 0]
        if len(filtered_batch) < len(batch):
             logger.warning(f"Collate function filtered {len(batch) - len(filtered_batch)} items due to missing/empty 'pixel_values'.")
        if not filtered_batch:
             return None # Return None if the whole batch was filtered
        try:
             # Only pass the 'pixel_values' key to the default collate function
             # Assumes 'pixel_values' contains the tensor we need
             pixel_values_only = [item['pixel_values'] for item in filtered_batch]
             collated_pixels = collate_to_use(pixel_values_only)
             # Return in the expected dictionary format for the training loop
             return {"pixel_values": collated_pixels}
        except Exception as e:
             logger.error(f"Error during collate_fn: {e}. Skipping batch.")
             # Return None or an empty dict to signal skipping the batch in the training loop
             return None


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not is_iterable, # Only shuffle map-style datasets
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate # Use the safe collate function
    )

if __name__ == '__main__':
    # Example Usage:
    logging.basicConfig(level=logging.INFO) # Setup basic logging for testing

    try:
        logger.info("\n--- Testing with ImageNette (320px config) ---")
        processed_imagenette = load_and_preprocess_dataset(
            dataset_name="frgfm/imagenette",
            dataset_config_name="320px", # Specify the config
            image_column="image",
            resolution=256, # Resize to 256
            max_samples=50, # Small subset for testing
            split="train"
        )
        print(f"Processed dataset type: {type(processed_imagenette)}")
        imagenette_loader = create_dataloader(processed_imagenette, batch_size=4, num_workers=0)
        imagenette_batch = next(iter(imagenette_loader))
        # Check if batch is None (meaning it might have been filtered)
        if imagenette_batch is not None and "pixel_values" in imagenette_batch:
            print(f"ImageNette Batch pixel_values shape: {imagenette_batch['pixel_values'].shape}")
        else:
             print("ImageNette loader returned an empty or invalid first batch.")


        logger.info("\n--- Testing with CIFAR-10 (no config needed) ---")
        processed_cifar = load_and_preprocess_dataset(
            dataset_name="uoft-cs/cifar10",
            dataset_config_name=None, # Explicitly None
            image_column="img",
            resolution=64,
            max_samples=50,
            split="test"
        )
        cifar_loader = create_dataloader(processed_cifar, batch_size=4, num_workers=0)
        cifar_batch = next(iter(cifar_loader))
        if cifar_batch is not None and "pixel_values" in cifar_batch:
             print(f"CIFAR-10 Batch pixel_values shape: {cifar_batch['pixel_values'].shape}")
        else:
             print("CIFAR-10 loader returned an empty or invalid first batch.")


    except Exception as e:
        logger.error(f"An error occurred during example usage: {e}", exc_info=True)

