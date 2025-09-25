"""Data loading utilities."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import torch


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    dataset_name: str = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    dataset_path: Optional[str] = None  # Local path to dataset
    task_name: Optional[str] = None  # Specific task within dataset
    train_split: str = "train"
    eval_split: str = "test"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    cache_dir: Optional[str] = None
    num_proc: int = 4  # Number of processes for preprocessing
    
    # Sampling parameters
    train_samples: Optional[int] = None  # Limit training samples
    eval_samples: Optional[int] = None  # Limit evaluation samples
    seed: int = 42


class DataLoader:
    """Data loader for genomic datasets."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.dataset: Optional[DatasetDict] = None
        
    def load(self) -> DatasetDict:
        """
        Load the dataset.
        
        Returns:
            DatasetDict with train and eval splits
        """
        # Determine dataset source
        if self.config.dataset_path and os.path.exists(self.config.dataset_path):
            print(f"Loading dataset from local path: {self.config.dataset_path}")
            self.dataset = load_from_disk(self.config.dataset_path)
        else:
            print(f"Loading dataset from HuggingFace: {self.config.dataset_name}")
            self.dataset = load_dataset(
                self.config.dataset_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
        # Filter by task if specified
        if self.config.task_name:
            self.dataset = self._filter_by_task(self.dataset, self.config.task_name)
            
        # Sample if requested
        if self.config.train_samples or self.config.eval_samples:
            self.dataset = self._sample_dataset(self.dataset)
            
        print(f"Dataset loaded successfully!")
        self._print_dataset_info()
        
        return self.dataset
    
    def preprocess(self, 
                   tokenizer: PreTrainedTokenizer,
                   label_column: str = "label") -> DatasetDict:
        """
        Preprocess the dataset for training.
        
        Args:
            tokenizer: Tokenizer to use for preprocessing
            label_column: Name of the label column
            
        Returns:
            Preprocessed DatasetDict
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
            
        print("Preprocessing dataset...")
        
        def preprocess_function(examples):
            """Tokenize the sequences."""
            # Handle different input column names
            if 'sequence' in examples:
                texts = examples['sequence']
            elif 'text' in examples:
                texts = examples['text']
            else:
                # Try to find the text column
                for key in examples.keys():
                    if isinstance(examples[key][0], str):
                        texts = examples[key]
                        break
                else:
                    raise ValueError("Could not find text column in dataset")
            
            # Tokenize
            result = tokenizer(
                texts,
                truncation=self.config.truncation,
                padding=self.config.padding,
                max_length=self.config.max_length
            )
            
            # Add labels if present
            if label_column in examples:
                result['labels'] = examples[label_column]
                
            return result
        
        # Apply preprocessing
        processed_dataset = self.dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.config.num_proc,
            remove_columns=self.dataset[self.config.train_split].column_names
        )
        
        # Set format for PyTorch
        processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        print("Preprocessing completed!")
        return processed_dataset
    
    def create_data_collator(self, tokenizer: PreTrainedTokenizer):
        """
        Create a data collator for the dataset.
        
        Args:
            tokenizer: Tokenizer to use for collation
            
        Returns:
            DataCollator instance
        """
        from transformers import DataCollatorWithPadding
        
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=self.config.max_length
        )
    
    def get_splits(self) -> Tuple[Dataset, Dataset]:
        """
        Get train and eval splits.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
            
        train_dataset = self.dataset.get(self.config.train_split)
        eval_dataset = self.dataset.get(self.config.eval_split)
        
        if train_dataset is None:
            raise ValueError(f"Train split '{self.config.train_split}' not found in dataset")
        if eval_dataset is None:
            raise ValueError(f"Eval split '{self.config.eval_split}' not found in dataset")
            
        return train_dataset, eval_dataset
    
    def _filter_by_task(self, dataset: DatasetDict, task_name: str) -> DatasetDict:
        """Filter dataset by task name."""
        print(f"Filtering dataset for task: {task_name}")
        
        filtered_splits = {}
        for split_name, split_data in dataset.items():
            if 'task' in split_data.column_names:
                filtered_splits[split_name] = split_data.filter(
                    lambda x: x['task'] == task_name,
                    num_proc=self.config.num_proc
                )
            else:
                print(f"Warning: 'task' column not found in {split_name} split")
                filtered_splits[split_name] = split_data
                
        return DatasetDict(filtered_splits)
    
    def _sample_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Sample the dataset to reduce size."""
        sampled_splits = {}
        
        for split_name, split_data in dataset.items():
            if split_name == self.config.train_split and self.config.train_samples:
                n_samples = min(self.config.train_samples, len(split_data))
                sampled_splits[split_name] = split_data.shuffle(seed=self.config.seed).select(range(n_samples))
                print(f"Sampled {n_samples} examples from {split_name} split")
                
            elif split_name == self.config.eval_split and self.config.eval_samples:
                n_samples = min(self.config.eval_samples, len(split_data))
                sampled_splits[split_name] = split_data.shuffle(seed=self.config.seed).select(range(n_samples))
                print(f"Sampled {n_samples} examples from {split_name} split")
                
            else:
                sampled_splits[split_name] = split_data
                
        return DatasetDict(sampled_splits)
    
    def _print_dataset_info(self) -> None:
        """Print information about the loaded dataset."""
        if self.dataset is None:
            return
            
        print("\nDataset Information:")
        print("-" * 40)
        
        for split_name, split_data in self.dataset.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Number of examples: {len(split_data)}")
            print(f"  Features: {list(split_data.features.keys())}")
            
            # Show sample if available
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"  Sample:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")
                        
        print("-" * 40)
    
    def save_to_disk(self, output_path: str) -> None:
        """
        Save the dataset to disk.
        
        Args:
            output_path: Path to save the dataset
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
            
        print(f"Saving dataset to: {output_path}")
        self.dataset.save_to_disk(output_path)
        print("Dataset saved successfully!")
    
    @staticmethod
    def download_dataset(dataset_name: str, output_path: str) -> None:
        """
        Download a dataset from HuggingFace Hub.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
            output_path: Local path to save the dataset
        """
        print(f"Downloading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        
        print(f"Saving to: {output_path}")
        dataset.save_to_disk(output_path)
        
        print(f"✅ Dataset downloaded and saved successfully")
