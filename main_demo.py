"""
Main demonstration script showing all three name generation models.
"""

import torch
from data_preprocessing import load_names, create_vocabulary, create_char_mappings
from bigram_model import BigramModel
from trigram_model import TrigramModel
from rnn_model import CustomRNN, RNNNameGenerator


def demonstrate_all_models():
    """Demonstrate all three name generation models."""
    
    print("=" * 60)
    print("NAME GENERATION MODELS DEMONSTRATION")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    names = load_names()
    vocabulary = create_vocabulary(names)
    char_to_int, int_to_char = create_char_mappings(vocabulary)
    
    print(f"   Loaded {len(names)} names")
    print(f"   Vocabulary size: {len(vocabulary)}")
    print(f"   Vocabulary: {vocabulary}")
    
    # Demonstrate Bigram Model
    print("\n2. BIGRAM MODEL")
    print("-" * 30)
    bigram_model = BigramModel(names, vocabulary, char_to_int)
    bigram_names = bigram_model.generate_unique_names('Joe', 10)
    print("Generated names:")
    for i, name in enumerate(bigram_names, 1):
        print(f"   {i:2d}. {name}")
    
    # Demonstrate Trigram Model
    print("\n3. TRIGRAM MODEL")
    print("-" * 30)
    trigram_model = TrigramModel(names, vocabulary, char_to_int)
    trigram_names = trigram_model.generate_unique_names('Joe', 10)
    print("Generated names:")
    for i, name in enumerate(trigram_names, 1):
        print(f"   {i:2d}. {name}")
    
    # Demonstrate RNN Model (requires training first)
    print("\n4. RNN MODEL")
    print("-" * 30)
    print("Note: RNN model needs to be trained first using train_rnn.py")
    print("Attempting to load pre-trained model...")
    
    try:
        # Try to load pre-trained model
        EMBED_SIZE = 32
        HIDDEN_DIM = 32
        MAX_NAME_LENGTH = 32
        
        rnn_model = CustomRNN(len(vocabulary), EMBED_SIZE, HIDDEN_DIM, MAX_NAME_LENGTH)
        rnn_model.load_state_dict(torch.load('trained_rnn_model.pth'))
        rnn_model.eval()
        
        generator = RNNNameGenerator(rnn_model, vocabulary, char_to_int, int_to_char, names)
        rnn_names = generator.generate_unique_names('Joe', 10)
        print("Generated names:")
        for i, name in enumerate(rnn_names, 1):
            print(f"   {i:2d}. {name}")
            
    except FileNotFoundError:
        print("   Pre-trained model not found. Please run train_rnn.py first.")
        print("   Creating untrained model for demonstration...")
        
        EMBED_SIZE = 32
        HIDDEN_DIM = 32
        MAX_NAME_LENGTH = 32
        
        rnn_model = CustomRNN(len(vocabulary), EMBED_SIZE, HIDDEN_DIM, MAX_NAME_LENGTH)
        generator = RNNNameGenerator(rnn_model, vocabulary, char_to_int, int_to_char, names)
        rnn_names = generator.generate_unique_names('Joe', 5)
        print("Generated names (untrained model - will be random):")
        for i, name in enumerate(rnn_names, 1):
            print(f"   {i:2d}. {name}")
    
    # Compare different starting strings
    print("\n5. COMPARISON WITH DIFFERENT STARTING STRINGS")
    print("-" * 50)
    
    start_strings = ['', 'A', 'Jo', 'Sam']
    
    for start in start_strings:
        print(f"\nStarting with '{start}':")
        
        # Bigram
        bigram_name = bigram_model.generate_name(start)
        print(f"   Bigram:  {bigram_name}")
        
        # Trigram
        trigram_name = trigram_model.generate_name(start)
        print(f"   Trigram: {trigram_name}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_all_models()
