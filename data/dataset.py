"""
Dataset utilities for HOI Spatial Linking Training.

Provides dataset loading and processing functions compatible with
HuggingFace datasets and TRL SFTTrainer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, load_dataset, concatenate_datasets
import logging

logger = logging.getLogger(__name__)


def load_hoi_dataset(
    data_files: Union[str, List[str]],
    split: str = "train",
    streaming: bool = False,
) -> Dataset:
    """
    Load HOI SFT dataset from JSON files.
    
    Args:
        data_files: Path(s) to JSON data files
        split: Dataset split name
        streaming: Whether to use streaming mode
        
    Returns:
        HuggingFace Dataset
    """
    if isinstance(data_files, str):
        data_files = [data_files]
    
    # Filter to existing files
    existing_files = [f for f in data_files if Path(f).exists()]
    
    if not existing_files:
        raise FileNotFoundError(f"No data files found: {data_files}")
    
    logger.info(f"Loading dataset from {len(existing_files)} files")
    
    dataset = load_dataset(
        "json",
        data_files=existing_files,
        split=split,
        streaming=streaming,
    )
    
    return dataset


def load_combined_dataset(
    spatial_sft_file: str,
    cof_sft_file: str,
    spatial_weight: float = 0.5,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Load and combine spatial linking and CoF tool-calling datasets.
    
    Args:
        spatial_sft_file: Path to spatial linking SFT data
        cof_sft_file: Path to CoF tool-calling SFT data
        spatial_weight: Weight for spatial samples (for weighted sampling)
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed
        
    Returns:
        Combined HuggingFace Dataset
    """
    datasets_to_combine = []
    
    if Path(spatial_sft_file).exists():
        spatial_ds = load_hoi_dataset(spatial_sft_file)
        # Add source column
        spatial_ds = spatial_ds.map(
            lambda x: {"source": "spatial", "weight": spatial_weight}
        )
        datasets_to_combine.append(spatial_ds)
        logger.info(f"Loaded spatial SFT: {len(spatial_ds)} samples")
    
    if Path(cof_sft_file).exists():
        cof_ds = load_hoi_dataset(cof_sft_file)
        cof_ds = cof_ds.map(
            lambda x: {"source": "cof", "weight": 1.0 - spatial_weight}
        )
        datasets_to_combine.append(cof_ds)
        logger.info(f"Loaded CoF SFT: {len(cof_ds)} samples")
    
    if not datasets_to_combine:
        raise FileNotFoundError("No dataset files found")
    
    # Combine
    combined = concatenate_datasets(datasets_to_combine)
    
    if shuffle:
        combined = combined.shuffle(seed=seed)
    
    logger.info(f"Combined dataset: {len(combined)} samples")
    
    return combined


class HOIDataset:
    """
    Wrapper class for HOI datasets with additional utilities.
    
    Provides:
    - Easy loading from various sources
    - Filtering by task type
    - Statistics and analysis
    """
    
    def __init__(
        self,
        data: Union[str, List[Dict], Dataset],
        name: str = "hoi_dataset",
    ):
        self.name = name
        
        if isinstance(data, str):
            # Load from file
            self.dataset = load_hoi_dataset(data)
        elif isinstance(data, list):
            # Create from list of dicts
            self.dataset = Dataset.from_list(data)
        else:
            self.dataset = data
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]
    
    def filter_by_task(self, task_type: str) -> "HOIDataset":
        """Filter to samples of a specific task type."""
        filtered = self.dataset.filter(
            lambda x: x.get("task_type") == task_type
        )
        return HOIDataset(filtered, name=f"{self.name}_{task_type}")
    
    def filter_by_tool_use(self, uses_tool: bool) -> "HOIDataset":
        """Filter to samples that use/don't use tools."""
        filtered = self.dataset.filter(
            lambda x: x.get("uses_tool", False) == uses_tool
        )
        name_suffix = "with_tools" if uses_tool else "no_tools"
        return HOIDataset(filtered, name=f"{self.name}_{name_suffix}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.dataset),
            "task_types": {},
            "tool_usage": {"with_tools": 0, "without_tools": 0},
            "datasets": {},
        }
        
        for sample in self.dataset:
            # Task type
            task = sample.get("task_type", "unknown")
            stats["task_types"][task] = stats["task_types"].get(task, 0) + 1
            
            # Tool usage
            if sample.get("uses_tool", False):
                stats["tool_usage"]["with_tools"] += 1
            else:
                stats["tool_usage"]["without_tools"] += 1
            
            # Dataset source
            metadata = sample.get("metadata", {})
            dataset = metadata.get("dataset", "unknown")
            stats["datasets"][dataset] = stats["datasets"].get(dataset, 0) + 1
        
        return stats
    
    def to_hf_dataset(self) -> Dataset:
        """Return the underlying HuggingFace Dataset."""
        return self.dataset
    
    def save(self, output_path: str):
        """Save dataset to JSON file."""
        data = [self.dataset[i] for i in range(len(self.dataset))]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} samples to {output_path}")


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Input dataset
        val_ratio: Ratio for validation set
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]
