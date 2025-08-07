"""
Data processing and preprocessing utilities for sentiment analysis.

Includes text tokenization, vocabulary building, data augmentation,
and batch processing for different datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SentimentVocabulary:
    """
    Vocabulary management for sentiment analysis.
    
    Handles token-to-ID mapping with special tokens for padding, unknown words, etc.
    """
    
    def __init__(
        self,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        cls_token: str = "<CLS>",
        sep_token: str = "<SEP>",
        min_freq: int = 2,
        max_vocab_size: int = 30000
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
        # Initialize special tokens
        self.token_to_id = {
            pad_token: 0,
            unk_token: 1,
            cls_token: 2,
            sep_token: 3
        }
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.token_freq = defaultdict(int)
        
        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = 2
        self.sep_id = 3
        
    def build_from_texts(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        logger.info(f"Building vocabulary from {len(texts)} texts...")
        
        # Count token frequencies
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.token_freq[token] += 1
        
        # Sort tokens by frequency and add to vocabulary
        sorted_tokens = sorted(
            self.token_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add tokens that meet frequency threshold
        vocab_size = len(self.token_to_id)  # Start after special tokens
        for token, freq in sorted_tokens:
            if freq >= self.min_freq and vocab_size < self.max_vocab_size:
                if token not in self.token_to_id:
                    self.token_to_id[token] = vocab_size
                    self.id_to_token[vocab_size] = token
                    vocab_size += 1
        
        logger.info(f"Built vocabulary with {len(self.token_to_id)} tokens")
        logger.info(f"Most frequent tokens: {sorted_tokens[:10]}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization - splits on whitespace and punctuation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and handle basic cleaning
        text = text.lower().strip()
        
        # Replace URLs with special token
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
        
        # Replace mentions with special token
        text = re.sub(r'@[A-Za-z0-9_]+', '<MENTION>', text)
        
        # Replace hashtags with special token
        text = re.sub(r'#[A-Za-z0-9_]+', '<HASHTAG>', text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r"\\b\\w+\\b|[.,!?;]", text)
        
        return tokens
    
    def encode(self, text: str, max_length: int = 128, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            add_special_tokens: Whether to add CLS and SEP tokens
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Convert to IDs
        input_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]
        
        # Truncate or pad
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids.extend([self.pad_id] * (max_length - len(input_ids)))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        tokens = [
            self.id_to_token.get(token_id, self.unk_token)
            for token_id in token_ids
            if token_id != self.pad_id
        ]
        
        # Remove special tokens for readable output
        tokens = [
            token for token in tokens
            if token not in [self.cls_token, self.sep_token, self.pad_token]
        ]
        
        return " ".join(tokens)
    
    def save(self, path: str) -> None:
        """Save vocabulary to disk."""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'token_freq': dict(self.token_freq),
            'config': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'cls_token': self.cls_token,
                'sep_token': self.sep_token,
                'min_freq': self.min_freq,
                'max_vocab_size': self.max_vocab_size
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SentimentVocabulary':
        """Load vocabulary from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Create instance with saved config
        config = vocab_data['config']
        vocab = cls(**config)
        
        # Restore vocabulary mappings
        vocab.token_to_id = vocab_data['token_to_id']
        vocab.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        vocab.token_freq = defaultdict(int, vocab_data['token_freq'])
        
        logger.info(f"Vocabulary loaded from {path}")
        return vocab
    
    def __len__(self) -> int:
        return len(self.token_to_id)


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    
    Handles text encoding, label conversion, and batch processing.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: SentimentVocabulary,
        max_length: int = 128,
        label_map: Optional[Dict[str, int]] = None
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.label_map = label_map or {'negative': 0, 'neutral': 1, 'positive': 2}
        
        assert len(texts) == len(labels), "Number of texts and labels must match"
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.vocab.encode(text, max_length=self.max_length)
        
        # Convert label to tensor
        encoded['labels'] = torch.tensor(label, dtype=torch.long)
        
        return encoded
    
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        text_column: str,
        label_column: str,
        vocab: Optional[SentimentVocabulary] = None,
        max_length: int = 128,
        label_map: Optional[Dict[str, int]] = None
    ) -> 'SentimentDataset':
        """
        Create dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            vocab: Vocabulary instance (will be created if None)
            max_length: Maximum sequence length
            label_map: Mapping from label strings to integers
            
        Returns:
            SentimentDataset instance
        """
        df = pd.read_csv(csv_path)
        
        texts = df[text_column].fillna("").astype(str).tolist()
        label_strings = df[label_column].fillna("neutral").astype(str).tolist()
        
        # Create or use provided label map
        if label_map is None:
            unique_labels = sorted(set(label_strings))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert string labels to integers
        labels = [label_map.get(label_str, 1) for label_str in label_strings]  # Default to neutral
        
        # Create vocabulary if not provided
        if vocab is None:
            vocab = SentimentVocabulary()
            vocab.build_from_texts(texts)
        
        logger.info(f"Created dataset from {csv_path}: {len(texts)} samples")
        logger.info(f"Label distribution: {Counter(labels)}")
        
        return cls(texts, labels, vocab, max_length, label_map)
    
    @classmethod
    def from_json(
        cls,
        json_path: str,
        text_field: str,
        label_field: str,
        vocab: Optional[SentimentVocabulary] = None,
        max_length: int = 128,
        label_map: Optional[Dict[str, int]] = None
    ) -> 'SentimentDataset':
        """Create dataset from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item[text_field] for item in data]
        label_strings = [item[label_field] for item in data]
        
        # Create label map
        if label_map is None:
            unique_labels = sorted(set(label_strings))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        labels = [label_map.get(label_str, 1) for label_str in label_strings]
        
        # Create vocabulary if not provided
        if vocab is None:
            vocab = SentimentVocabulary()
            vocab.build_from_texts(texts)
        
        logger.info(f"Created dataset from {json_path}: {len(texts)} samples")
        
        return cls(texts, labels, vocab, max_length, label_map)


class SentimentDataAugmentation:
    """
    Data augmentation techniques for sentiment analysis.
    
    Includes synonym replacement, random insertion, random swap, and random deletion.
    """
    
    def __init__(self, augmentation_prob: float = 0.1):
        self.augmentation_prob = augmentation_prob
        
        # Simple synonym dictionary - in practice, would use WordNet or similar
        self.synonyms = {
            'good': ['great', 'excellent', 'wonderful', 'amazing', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'glad'],
            'sad': ['unhappy', 'depressed', 'miserable', 'gloomy', 'sorrowful'],
            'love': ['adore', 'cherish', 'treasure', 'appreciate', 'enjoy'],
            'hate': ['despise', 'detest', 'loathe', 'dislike', 'abhor']
        }
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms."""
        words = text.split()
        if len(words) < 1:
            return text
        
        new_words = words.copy()
        random_word_list = list(set([word.lower() for word in words]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            if random_word in self.synonyms and num_replaced < n:
                synonym = random.choice(self.synonyms[random_word])
                for i, word in enumerate(new_words):
                    if word.lower() == random_word:
                        new_words[i] = synonym
                        break
                num_replaced += 1
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random words from the text."""
        words = text.split()
        if len(words) < 1:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, random_word)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Swap n pairs of words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Delete words with probability p."""
        words = text.split()
        if len(words) < 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        
        # If all words are deleted, return a random word
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment(self, text: str, num_aug: int = 4) -> List[str]:
        """Apply multiple augmentation techniques."""
        augmented_texts = [text]  # Include original
        
        if random.random() < self.augmentation_prob:
            # Apply different augmentation techniques
            techniques = [
                self.synonym_replacement,
                self.random_insertion,
                self.random_swap,
                self.random_deletion
            ]
            
            for i in range(min(num_aug, len(techniques))):
                aug_text = techniques[i](text)
                if aug_text != text:  # Only add if actually changed
                    augmented_texts.append(aug_text)
        
        return augmented_texts


def create_data_loaders(
    texts: List[str],
    labels: List[int],
    vocab: SentimentVocabulary,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    augment_data: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        vocab: Vocabulary for encoding
        batch_size: Batch size for data loaders
        test_size: Proportion of data for testing
        val_size: Proportion of remaining data for validation
        max_length: Maximum sequence length
        shuffle: Whether to shuffle training data
        num_workers: Number of workers for data loading
        augment_data: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split into train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Split train into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=42, stratify=train_labels
    )
    
    # Apply data augmentation if requested
    if augment_data:
        logger.info("Applying data augmentation to training set...")
        augmenter = SentimentDataAugmentation()
        
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(train_texts, train_labels):
            aug_texts = augmenter.augment(text)
            augmented_texts.extend(aug_texts)
            augmented_labels.extend([label] * len(aug_texts))
        
        train_texts = augmented_texts
        train_labels = augmented_labels
        
        logger.info(f"Training set size after augmentation: {len(train_texts)}")
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def load_standard_datasets(dataset_name: str, data_dir: str = "./data") -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Load standard sentiment analysis datasets.
    
    Args:
        dataset_name: Name of dataset ('imdb', 'amazon', 'yelp', 'twitter')
        data_dir: Directory containing datasets
        
    Returns:
        Tuple of (texts, labels, label_map)
    """
    data_path = Path(data_dir) / dataset_name
    
    if dataset_name.lower() == 'imdb':
        # Load IMDB dataset format
        texts, labels = [], []
        label_map = {'negative': 0, 'positive': 1}
        
        for label_name, label_id in label_map.items():
            label_dir = data_path / label_name
            if label_dir.exists():
                for file_path in label_dir.glob('*.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        texts.append(text)
                        labels.append(label_id)
    
    elif dataset_name.lower() in ['amazon', 'yelp']:
        # Load review datasets from CSV
        csv_file = data_path / f"{dataset_name.lower()}_reviews.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            texts = df['review_text'].fillna("").astype(str).tolist()
            
            # Convert ratings to sentiment labels
            ratings = df['rating'].tolist()
            labels = []
            for rating in ratings:
                if rating <= 2:
                    labels.append(0)  # negative
                elif rating == 3:
                    labels.append(1)  # neutral
                else:
                    labels.append(2)  # positive
            
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    elif dataset_name.lower() == 'twitter':
        # Load Twitter sentiment dataset
        json_file = data_path / "twitter_sentiment.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = [item['text'] for item in data]
            label_strings = [item['sentiment'] for item in data]
            
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            labels = [label_map.get(label_str, 1) for label_str in label_strings]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Loaded {dataset_name} dataset: {len(texts)} samples")
    logger.info(f"Label distribution: {Counter(labels)}")
    
    return texts, labels, label_map