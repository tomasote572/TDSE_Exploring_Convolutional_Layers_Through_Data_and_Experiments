"""
Example Usage of Sign Language MNIST CNN Classifier

This script demonstrates how to use the prediction function programmatically.
"""

from PIL import Image
import numpy as np
from app import predict

# Example 1: Create a random test image
print("Example 1: Random image prediction")
random_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
result = predict(random_img)
print(f"Prediction: {result}\n")

# Example 2: Create a simple white image
print("Example 2: White image prediction")
white_img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
result = predict(white_img)
print(f"Prediction: {result}\n")

# Example 3: Create a black image
print("Example 3: Black image prediction")
black_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
result = predict(black_img)
print(f"Prediction: {result}\n")

# Example 4: Handle None input
print("Example 4: None input handling")
result = predict(None)
print(f"Result: {result}\n")

print("=" * 60)
print("To use the interactive Gradio interface, run:")
print("  python app.py")
print("=" * 60)
