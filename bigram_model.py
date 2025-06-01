"""
Bigram language model for name generation.
"""

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_names, create_vocabulary, create_char_mappings


class BigramModel:
    def __init__(self, names, vocabulary, char_to_int):
        self.names = names
        self.vocabulary = vocabulary
        self.char_to_int = char_to_int
        self.lookup_table = self._build_lookup_table()
    
    def _build_lookup_table(self):
        """Build the bigram lookup table."""
        lookup_table = torch.zeros((len(self.vocabulary), len(self.vocabulary)), dtype=torch.int32)
        
        for name in self.names:
            for i in range(len(name) - 1):
                ix1 = self.char_to_int[name[i]]
                ix2 = self.char_to_int[name[i+1]]
                lookup_table[ix1, ix2] += 1
        
        return lookup_table
    
    def visualize_bigrams(self):
        """Visualize the bigram counts."""
        x = [i for i in range(len(self.lookup_table)) for _ in range(len(self.lookup_table[0]))]
        y = [j for _ in range(len(self.lookup_table)) for j in range(len(self.lookup_table[0]))]
        counts = [self.lookup_table[i][j] for i in range(len(self.lookup_table)) for j in range(len(self.lookup_table[0]))]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x, y, s=counts, c=counts, cmap='Reds', alpha=0.7)
        plt.xticks(ticks=np.arange(len(self.vocabulary)), labels=self.vocabulary)
        plt.yticks(ticks=np.arange(len(self.vocabulary)), labels=self.vocabulary)
        plt.xlabel('First Character')
        plt.ylabel('Second Character')
        plt.title('Bigram Counts')
        plt.colorbar(scatter, label='Count')
        plt.grid(True)
        plt.show()
    
    def generate_name(self, start_string=''):
        """Generate a name using the bigram model."""
        name = '<' + start_string.lower()
        
        while True:
            ix1 = self.char_to_int[name[-1]]
            next_char_probs = self.lookup_table[ix1]
            total_weight = sum(next_char_probs)
            
            if total_weight > 0:
                next_char = random.choices(self.vocabulary, weights=next_char_probs, k=1)[0]
            else:
                next_char = random.choice(self.vocabulary)
            
            if next_char == '>':
                break
            name += next_char
        
        return name[1:].capitalize()
    
    def generate_unique_names(self, start_string='', n=10):
        """Generate n unique names that are not in the training set."""
        unique_names = set()
        
        while len(unique_names) < n:
            name = self.generate_name(start_string)
            if '<' + name.lower() + '>' not in self.names:
                unique_names.add(name)
        
        return list(unique_names)


if __name__ == "__main__":
    # Load data
    names = load_names()
    vocabulary = create_vocabulary(names)
    char_to_int, int_to_char = create_char_mappings(vocabulary)
    
    # Create and use bigram model
    bigram_model = BigramModel(names, vocabulary, char_to_int)
    
    # Visualize bigrams
    bigram_model.visualize_bigrams()
    
    # Generate names
    generated_names = bigram_model.generate_unique_names('Joe', 10)
    print("Generated names using Bigram Model:")
    for name in generated_names:
        print(name)
