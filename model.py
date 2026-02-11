"""
CNN Model for Sign Language MNIST Classification
This module defines the convolutional neural network architecture.
"""
import torch
import torch.nn as nn


class SignLanguageCNN(nn.Module):
    """
    Convolutional Neural Network for Sign Language MNIST classification.
    
    Architecture:
    - Conv2d layer with 32 filters (5x5 kernel)
    - ReLU activation
    - MaxPool2d (2x2)
    - Conv2d layer with 64 filters (5x5 kernel)
    - ReLU activation
    - MaxPool2d (2x2)
    - Flatten
    - Fully connected layer (128 units)
    - ReLU activation
    - Dropout (0.5)
    - Output layer (25 classes for letters A-Y, excluding J and Z)
    """
    
    def __init__(self):
        super(SignLanguageCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After two pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 25)  # 25 classes (A-Y, excluding J and Z)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, 25)
        """
        # First conv block
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model():
    """
    Create and return a new instance of the Sign Language CNN model.
    
    Returns:
        SignLanguageCNN: A new model instance
    """
    model = SignLanguageCNN()
    return model


def load_trained_model(path='sign_language_cnn.pth'):
    """
    Load a pre-trained model from disk.
    
    Args:
        path: Path to the saved model weights
        
    Returns:
        SignLanguageCNN: Model with loaded weights
    """
    model = create_model()
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Warning: Model file '{path}' not found. Returning untrained model.")
        return model


# Create a global model instance for use in the Gradio app
cnn_model = create_model()
