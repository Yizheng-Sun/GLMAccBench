"""Data loading utilities for genomic datasets."""

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
    max_length: Optional[int] = None  # If None, will auto-detect from data
    truncation: bool = True
    padding: bool = False  # Changed to False to match train_nucleotide_transformer.py
    cache_dir: Optional[str] = None
    num_proc: int = 4  # Number of processes for preprocessing
    
    # Train/validation split parameters
    val_split_ratio: float = 0.2  # 20% for validation
    stratify_by_label: bool = True  # Try to stratify by label
    seed: int = 42
    
    # Sampling parameters
    train_samples: Optional[int] = None  # Limit training samples
    eval_samples: Optional[int] = None  # Limit evaluation samples


class SimpleDataCollator:
    """Simple data collator that creates tensors without padding."""
    
    def __call__(self, features):
        """Collate features into a batch."""
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long)
        }
        return batch


class DataLoader:
    """Data loader for genomic datasets following train_nucleotide_transformer.py pattern."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.raw_dataset: Optional[Dict] = None
        self.dataset: Optional[DatasetDict] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
    def load(self) -> Dict:
        """
        Load the raw dataset from local path or HuggingFace.
        
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        # Load from local path if it exists
        if self.config.dataset_path and os.path.exists(self.config.dataset_path):
            print(f"Loading dataset from local directory: {self.config.dataset_path}")
            
            # Try loading from arrow files first (preserves structure better)
            train_arrow = f'{self.config.dataset_path}/train/data-00000-of-00001.arrow'
            test_arrow = f'{self.config.dataset_path}/test/data-00000-of-00001.arrow'
            
            if os.path.exists(train_arrow) and os.path.exists(test_arrow):
                train_data = load_dataset('arrow', data_files=train_arrow)['train']
                test_data = load_dataset('arrow', data_files=test_arrow)['train']
                self.raw_dataset = {'train': train_data, 'test': test_data}
            else:
                # Fallback to load_from_disk
                dataset_dict = load_from_disk(self.config.dataset_path)
                self.raw_dataset = {
                    'train': dataset_dict['train'],
                    'test': dataset_dict['test']
                }
        else:
            # Load from HuggingFace Hub
            print(f"Loading dataset from HuggingFace: {self.config.dataset_name}")
            dataset = load_dataset(
                self.config.dataset_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            self.raw_dataset = {'train': dataset['train'], 'test': dataset['test']}
            print("Dataset downloaded. Consider saving it locally for faster future loading.")
        
        print(f"✅ Dataset loaded successfully!")
        self._print_dataset_info(self.raw_dataset)
        
        return self.raw_dataset
    
    def prepare_task_data(self, task_name: str, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset]:
        """
        Prepare train and validation datasets for a specific task.
        
        This follows the exact pattern from train_nucleotide_transformer.py:
        1. Filter by task name
        2. Split into train/val (80/20)
        3. Find max sequence length
        4. Tokenize without padding
        5. Remove unnecessary columns
        
        Args:
            task_name: Name of the task to train on
            tokenizer: Tokenizer to use for preprocessing
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        self.tokenizer = tokenizer
        print(f"\n===== Preparing data for task: {task_name} =====")
        
        # Filter for the specific task (use only the original train split)
        full_train = self.raw_dataset["train"].filter(lambda ex: ex["task"] == task_name)
        print(f"Found {len(full_train)} examples for task '{task_name}'")
        
        # Sample if requested
        if self.config.train_samples and self.config.train_samples < len(full_train):
            n_samples = self.config.train_samples
            full_train = full_train.shuffle(seed=self.config.seed).select(range(n_samples))
            print(f"Sampled {n_samples} examples from training data")
        
        # Split into train (80%) and validation (20%)
        if self.config.stratify_by_label:
            try:
                split = full_train.train_test_split(
                    test_size=self.config.val_split_ratio, 
                    seed=self.config.seed, 
                    stratify_by_column="label"
                )
                print(f"✓ Stratified split by label (train: {len(split['train'])}, val: {len(split['test'])})")
            except Exception as e:
                print(f"⚠️  Stratification failed ({e}), using random split")
                split = full_train.train_test_split(
                    test_size=self.config.val_split_ratio, 
                    seed=self.config.seed
                )
        else:
            split = full_train.train_test_split(
                test_size=self.config.val_split_ratio, 
                seed=self.config.seed
            )
        
        train = split["train"]
        val = split["test"]
        
        print(f"Train samples: {len(train)}, Validation samples: {len(val)}")
        
        # Find the maximum sequence length in the dataset
        if self.config.max_length is None:
            print("Finding maximum sequence length in the dataset...")
            train_max_length = max(len(seq) for seq in train["sequence"])
            val_max_length = max(len(seq) for seq in val["sequence"])
            actual_max_length = max(train_max_length, val_max_length)
            print(f"Max length in train: {train_max_length}, val: {val_max_length}")
            print(f"Using max_length: {actual_max_length}")
        else:
            actual_max_length = self.config.max_length
            print(f"Using configured max_length: {actual_max_length}")
        
        # Tokenize function
        def tokenize(batch):
            return tokenizer(
                batch["sequence"],
                padding=self.config.padding,  # False by default - no padding during tokenization
                truncation=self.config.truncation,
                max_length=actual_max_length,
                return_attention_mask=True,
            )
        
        print(f"Tokenizer model max length: {tokenizer.model_max_length}")
        
        # Tokenize datasets and remove all columns except 'label'
        print("Tokenizing training data...")
        train_tok = train.map(
            tokenize, 
            batched=True, 
            remove_columns=[c for c in train.column_names if c not in {"label"}]
        )
        
        print("Tokenizing validation data...")
        val_tok = val.map(
            tokenize, 
            batched=True, 
            remove_columns=[c for c in val.column_names if c not in {"label"}]
        )
        
        print("✅ Data preparation completed!")
        
        return train_tok, val_tok
    
    def get_test_data(self, task_name: str, tokenizer: PreTrainedTokenizer) -> Dataset:
        """
        Get the test dataset for a specific task.
        
        Args:
            task_name: Name of the task
            tokenizer: Tokenizer to use for preprocessing
            
        Returns:
            Tokenized test dataset
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        print(f"\nPreparing test data for task: {task_name}")
        
        # Filter test set by task
        test = self.raw_dataset["test"].filter(lambda ex: ex["task"] == task_name)
        print(f"Found {len(test)} test examples")
        
        # Sample if requested
        if self.config.eval_samples and self.config.eval_samples < len(test):
            n_samples = self.config.eval_samples
            test = test.shuffle(seed=self.config.seed).select(range(n_samples))
            print(f"Sampled {n_samples} examples from test data")
        
        # Find max length in test data
        test_max_length = max(len(seq) for seq in test["sequence"])
        print(f"Max sequence length in test data: {test_max_length}")
        
        # Tokenize function
        def tokenize(batch):
            return tokenizer(
                batch["sequence"],
                padding=self.config.padding,
                truncation=self.config.truncation,
                max_length=test_max_length,
                return_attention_mask=True,
            )
        
        # Tokenize and remove unnecessary columns
        test_tok = test.map(
            tokenize, 
            batched=True, 
            remove_columns=[c for c in test.column_names if c not in {"label"}]
        )
        
        print("✅ Test data preparation completed!")
        
        return test_tok
    
    def get_num_labels(self, task_name: str) -> int:
        """
        Get the number of labels for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Number of unique labels
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        # Filter by task and get unique labels
        task_data = self.raw_dataset["train"].filter(lambda ex: ex["task"] == task_name)
        num_labels = len(set(task_data["label"]))
        
        return num_labels
    
    def get_label_mappings(self, task_name: str) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        Get label to id and id to label mappings for a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Tuple of (id2label, label2id) dictionaries
        """
        num_labels = self.get_num_labels(task_name)
        id2label = {i: f"CLASS_{i}" for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}
        
        return id2label, label2id
    
    def get_available_tasks(self) -> list:
        """
        Get list of available tasks in the dataset.
        
        Returns:
            List of task names
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        # Get unique task names
        tasks = set(self.raw_dataset["train"]["task"])
        return sorted(list(tasks))
    
    def create_data_collator(self) -> SimpleDataCollator:
        """
        Create a data collator for the dataset.
        
        Returns:
            SimpleDataCollator instance
        """
        return SimpleDataCollator()
    
    def _print_dataset_info(self, dataset: Dict) -> None:
        """Print information about the loaded dataset."""
        print("\nDataset Information:")
        print("-" * 60)
        
        for split_name in ['train', 'test']:
            if split_name in dataset:
                split_data = dataset[split_name]
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
                
                # Show task distribution if 'task' column exists
                if 'task' in split_data.column_names:
                    tasks = set(split_data["task"])
                    print(f"  Available tasks: {sorted(list(tasks))}")
                    
        print("-" * 60)
    
    def save_to_disk(self, output_path: str) -> None:
        """
        Save the dataset to disk.
        
        Args:
            output_path: Path to save the dataset
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        print(f"Saving dataset to: {output_path}")
        
        # Convert dict to DatasetDict for saving
        dataset_dict = DatasetDict({
            'train': self.raw_dataset['train'],
            'test': self.raw_dataset['test']
        })
        dataset_dict.save_to_disk(output_path)
        
        print("✅ Dataset saved successfully!")
    
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
        
        print(f"✅ Dataset downloaded and saved successfully!")
