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
        image_column: Name of the column containing PIL images.
        resolution: Target resolution for image preprocessing.
        max_samples: Maximum number of samples to use (uses full dataset if None).
        split: Dataset split to load (e.g., 'train', 'test').
        streaming: Whether to stream the dataset (useful for large datasets).
        cache_dir: Optional directory for caching downloaded datasets.

    Returns:
        The processed Hugging Face Dataset object.

    Raises:
        ValueError: If the specified image column is not found.
        Exception: For errors during dataset loading.
    """
    logger.info(f"Loading dataset '{dataset_name}' (split: {split})...")
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )
        logger.info("Dataset loaded successfully.")
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
        if max_samples > len(dataset):
             logger.warning(f"max_samples ({max_samples}) > dataset size ({len(dataset)}). Using full dataset.")
        else:
            dataset = dataset.select(range(max_samples))
    elif max_samples is not None and streaming:
        logger.warning("max_samples is specified, but dataset is streamed. Taking the first max_samples.")
        dataset = dataset.take(max_samples)


    # Define the transformation function
    transform = get_transform(resolution)

    def transform_images(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the transform to a batch of examples."""
        try:
            # Handle potential variations in how images are loaded by datasets
            images_in = examples[image_column]
            if not isinstance(images_in, list):
                 images_in = [images_in] # Ensure it's a list

            processed_images = [transform(image) for image in images_in if isinstance(image, PILImage.Image)]

            # Handle cases where the column might not contain PIL images directly
            if not processed_images:
                 logger.warning(f"No processable PIL Images found in batch for column '{image_column}'. Skipping batch transformation.")
                 # Return original examples or handle appropriately
                 return examples # Or perhaps filter? Needs careful consideration

            return {"pixel_values": processed_images}
        except Exception as e:
            logger.error(f"Error transforming image: {e}. Skipping image.")
            # Return something to avoid breaking the pipeline, maybe None or skip the example
            # Returning pixel_values as empty list might be safer for batching
            return {"pixel_values": []}


    logger.info(f"Applying transforms (resolution: {resolution})...")
    # Use `with_transform` for efficiency (applies transform on the fly)
    # Note: Error handling within transform_images is basic. More robust handling might be needed.
    processed_dataset = dataset.with_transform(transform_images)
    logger.info("Dataset preprocessing complete.")

    # Filter out examples where transformation failed (resulted in empty pixel_values)
    # This filtering might be slow for large datasets, do only if necessary
    # def filter_empty(example):
    #     return example.get("pixel_values") is not None and len(example.get("pixel_values")) > 0
    # processed_dataset = processed_dataset.filter(filter_empty)


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
    if isinstance(dataset, torch.utils.data.IterableDataset) and shuffle:
        logger.warning("Shuffle=True may have limited effect with IterableDataset (streaming).")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not isinstance(dataset, torch.utils.data.IterableDataset), # Only shuffle map-style datasets
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn # Use default collate if None
    )

if __name__ == '__main__':
    # Example Usage:
    logging.basicConfig(level=logging.INFO) # Setup basic logging for testing

    try:
        # Test with a small standard dataset like CIFAR-10
        test_dataset_name = "uoft-cs/cifar10" # Official HF ID
        test_image_column = "img"
        test_resolution = 64 # Use smaller resolution for quick test
        test_max_samples = 100
        test_batch_size = 16

        logger.info("--- Testing Dataset Loading and Preprocessing ---")
        processed_data = load_and_preprocess_dataset(
            dataset_name=test_dataset_name,
            image_column=test_image_column,
            resolution=test_resolution,
            max_samples=test_max_samples,
            split="test" # Use test split for quick loading
        )
        print(f"Processed dataset type: {type(processed_data)}")
        # Accessing an element triggers the transform
        sample = processed_data[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample pixel_values type: {type(sample['pixel_values'])}")
        print(f"Sample pixel_values shape: {sample['pixel_values'].shape}")
        print(f"Sample pixel_values min/max: {sample['pixel_values'].min()}, {sample['pixel_values'].max()}")

        logger.info("\n--- Testing DataLoader Creation ---")
        dataloader = create_dataloader(
            processed_data,
            batch_size=test_batch_size,
            num_workers=0 # Use 0 for simple testing
        )
        print(f"DataLoader created: {dataloader}")

        # Fetch one batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch pixel_values type: {type(batch['pixel_values'])}")
        print(f"Batch pixel_values shape: {batch['pixel_values'].shape}") # Should be [batch_size, C, H, W]

        logger.info("\n--- Testing with Pokemon Dataset (if accessible) ---")
        try:
             processed_pokemon = load_and_preprocess_dataset(
                 dataset_name="lambdalabs/pokemon-blip-captions",
                 image_column="image",
                 resolution=256,
                 max_samples=50,
                 split="train"
             )
             poke_loader = create_dataloader(processed_pokemon, batch_size=4)
             poke_batch = next(iter(poke_loader))
             print(f"Pokemon Batch pixel_values shape: {poke_batch['pixel_values'].shape}")
        except Exception as poke_e:
             print(f"Could not test Pokemon dataset: {poke_e}")


    except Exception as e:
        logger.error(f"An error occurred during example usage: {e}", exc_info=True)

