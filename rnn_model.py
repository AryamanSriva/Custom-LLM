"""
Custom RNN model for name generation.
"""

import torch
import torch.nn as nn
from data_preprocessing import char_to_tensor


class CustomRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_dim, output_size):
        super(CustomRNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        x = self.embedding(x).view(1, 1, -1)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.view(1, -1))
        return out, hidden


class RNNNameGenerator:
    def __init__(self, model, vocabulary, char_to_int, int_to_char, names, max_length=32):
        self.model = model
        self.vocabulary = vocabulary
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char
        self.names = names
        self.max_length = max_length
    
    def generate_name(self, start_string='', temperature=0.8):
        """Generate text with the model."""
        start_string = '<' + start_string.lower()
        input_string = char_to_tensor(start_string, self.char_to_int)
        hidden = torch.zeros(1, 1, self.model.hidden_dim)

        for i in range(len(start_string) - 1):
            _, hidden = self.model(input_string[i].unsqueeze(0), hidden)

        generated_text = start_string[1:]
        input_char = input_string[-1].unsqueeze(0)

        for _ in range(self.max_length):
            output, hidden = self.model(input_char.unsqueeze(0), hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            next_char = self.int_to_char[top_i.item()]
            if next_char == '>':
                break
            generated_text += next_char
            input_char = char_to_tensor(next_char, self.char_to_int)
        
        return generated_text.capitalize()
    
    def generate_unique_names(self, start_string='', n=10):
        """Generate n unique names that are not in the training set."""
        unique_names = set()
        
        while len(unique_names) < n:
            name = self.generate_name(start_string)
            if '<' + name.lower() + '>' not in self.names:
                unique_names.add(name)
        
        return list(unique_names)


if __name__ == "__main__":
    from data_preprocessing import load_names, create_vocabulary, create_char_mappings
    
    # Load data
    names = load_names()
    vocabulary = create_vocabulary(names)
    char_to_int, int_to_char = create_char_mappings(vocabulary)
    
    # Model parameters
    EMBED_SIZE = 32
    HIDDEN_DIM = 32
    MAX_NAME_LENGTH = 32
    
    # Create model
    model = CustomRNN(len(vocabulary), EMBED_SIZE, HIDDEN_DIM, MAX_NAME_LENGTH)
    
    # Create generator
    generator = RNNNameGenerator(model, vocabulary, char_to_int, int_to_char, names, MAX_NAME_LENGTH)
    
    # Generate names (Note: model needs to be trained first)
    print("Generated names using RNN Model (untrained):")
    generated_names = generator.generate_unique_names('Joe', 5)
    for name in generated_names:
        print(name)
