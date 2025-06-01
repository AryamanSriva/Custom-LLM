"""
Training script for the Custom RNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from data_preprocessing import load_names, create_vocabulary, create_char_mappings, char_to_tensor
from rnn_model import CustomRNN, RNNNameGenerator


def train_rnn_model(names, vocabulary, char_to_int, int_to_char, 
                   embed_size=32, hidden_dim=32, max_name_length=32, 
                   epochs=3, learning_rate=0.005):
    """Train the RNN model."""
    
    # Training setup
    model = CustomRNN(len(vocabulary), embed_size, hidden_dim, max_name_length)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create generator for evaluation
    generator = RNNNameGenerator(model, vocabulary, char_to_int, int_to_char, names, max_name_length)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(names)  # Shuffle the dataset to ensure different order each epoch

        for name in names:
            inputs = char_to_tensor(name[:-1], char_to_int)
            targets = char_to_tensor(name[1:], char_to_int)
            
            hidden = torch.zeros(1, 1, hidden_dim)
            optimizer.zero_grad()
            loss = 0

            for i in range(len(inputs)):
                output, hidden = model(inputs[i].unsqueeze(0), hidden)
                loss += criterion(output, targets[i].unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / len(inputs)
        
        avg_loss = total_loss / len(names)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')
        
        # Generate sample names
        print("Sample generated names:")
        generated_names = generator.generate_unique_names('Joe', 10)
        for name in generated_names:
            print(f"  {name}")
        print('=' * 50)
    
    return model, losses


def plot_training_loss(losses, epochs):
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    names = load_names()
    vocabulary = create_vocabulary(names)
    char_to_int, int_to_char = create_char_mappings(vocabulary)
    
    print(f"Loaded {len(names)} names")
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Model parameters
    EMBED_SIZE = 32
    HIDDEN_DIM = 32
    MAX_NAME_LENGTH = 32
    EPOCHS = 3
    LEARNING_RATE = 0.005
    
    # Train the model
    print("Training RNN model...")
    trained_model, losses = train_rnn_model(
        names, vocabulary, char_to_int, int_to_char,
        embed_size=EMBED_SIZE,
        hidden_dim=HIDDEN_DIM,
        max_name_length=MAX_NAME_LENGTH,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Plot training loss
    plot_training_loss(losses, EPOCHS)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_rnn_model.pth')
    print("Model saved as 'trained_rnn_model.pth'")
    
    # Final generation test
    generator = RNNNameGenerator(trained_model, vocabulary, char_to_int, int_to_char, names)
    print("\nFinal generated names:")
    final_names = generator.generate_unique_names('Joe', 20)
    for name in final_names:
        print(name)
