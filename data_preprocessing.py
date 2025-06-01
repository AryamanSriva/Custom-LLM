"""
Data preprocessing utilities for name generation models.
Handles loading and preprocessing of text data.
"""

import torch


def load_names(filename='names.txt'):
    """Load names from a text file and add start/end tokens."""
    with open(filename, 'r') as file:
        names = file.read().splitlines()
    
    # Add start and end tokens
    names = ['<' + name.lower() + '>' for name in names]
    return names


def create_vocabulary(names):
    """Create vocabulary from the list of names."""
    vocabulary = set(''.join(names))
    vocabulary = ''.join(sorted(vocabulary))
    return vocabulary


def create_char_mappings(vocabulary):
    """Create character to integer and integer to character mappings."""
    char_to_int = {char: i for i, char in enumerate(vocabulary)}
    int_to_char = {idx: char for idx, char in enumerate(vocabulary)}
    return char_to_int, int_to_char


def char_to_tensor(text, char_to_int):
    """Convert a string of characters to a tensor of integer indexes."""
    return torch.tensor([char_to_int[char] for char in text], dtype=torch.long)


if __name__ == "__main__":
    # Example usage
    names = load_names()
    vocabulary = create_vocabulary(names)
    char_to_int, int_to_char = create_char_mappings(vocabulary)
    
    print(f"Loaded {len(names)} names")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Vocabulary: {vocabulary}")
