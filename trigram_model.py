"""
Trigram language model for name generation.
"""

import torch
import random
from data_preprocessing import load_names, create_vocabulary, create_char_mappings


class TrigramModel:
    def __init__(self, names, vocabulary, char_to_int):
        self.names = names
        self.vocabulary = vocabulary
        self.char_to_int = char_to_int
        self.lookup_table = self._build_lookup_table()
    
    def _build_lookup_table(self):
        """Build the trigram lookup table."""
        lookup_table = torch.zeros(
            (len(self.vocabulary), len(self.vocabulary), len(self.vocabulary)), 
            dtype=torch.int32
        )
        
        for name in self.names:
            for i in range(len(name) - 2):
                ix1 = self.char_to_int[name[i]]
                ix2 = self.char_to_int[name[i+1]]
                ix3 = self.char_to_int[name[i+2]]
                lookup_table[ix1, ix2, ix3] += 1
        
        return lookup_table
    
    def generate_name(self, start_string=''):
        """Generate a name using the trigram model."""
        name = '<' + start_string.lower()
        
        while True:
            if len(name) < 2:
                next_char = random.choice(self.vocabulary)
            else:
                ix1 = self.char_to_int[name[-2]]
                ix2 = self.char_to_int[name[-1]]
                next_char_probs = self.lookup_table[ix1, ix2]
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
    
    # Create and use trigram model
    trigram_model = TrigramModel(names, vocabulary, char_to_int)
    
    # Generate names
    generated_names = trigram_model.generate_unique_names('Joe', 10)
    print("Generated names using Trigram Model:")
    for name in generated_names:
        print(name)
