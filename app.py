"""
Gradio Interface for Sign Language MNIST Prediction

This application provides a web interface for predicting sign language letters
from hand gesture images using a trained CNN model.

What type of images can you use:
------------------------------------
This model is designed for Sign Language MNIST images with the following characteristics:

1. **Content**: Hand gestures representing American Sign Language (ASL) letters
   - Supports letters A-Y (excluding J and Z, which require motion)
   
2. **Format**: Any common image format (PNG, JPG, JPEG, etc.)

3. **Optimal characteristics**:
   - Grayscale or color (will be converted to grayscale)
   - Hand gesture clearly visible against a simple background
   - Hand centered in the image
   - Good lighting and contrast
   
4. **Image will be automatically processed**:
   - Converted to grayscale
   - Resized to 28x28 pixels
   - Normalized for the model

5. **Best results with**:
   - Clear, well-lit photos of hand gestures
   - Simple backgrounds
   - Hand clearly visible
   - Similar to ASL letter signs

Example usage: Upload or draw a hand gesture representing an ASL letter (A-Y).
"""

import gradio as gr
import torchvision.transforms as T
from PIL import Image
import torch
from model import cnn_model

# Use the pre-trained cnn_model
model = cnn_model.eval()

# Image transformation pipeline
transform = T.Compose([
    T.Grayscale(),      # Convert to grayscale (1 channel)
    T.Resize((28, 28)), # Resize to 28x28 pixels
    T.ToTensor()        # Convert to tensor and normalize to [0, 1]
])

# Mapping from class index to letter (Sign Language MNIST excludes J and Z)
CLASS_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}


def predict(img: Image.Image):
    """
    Predict the sign language letter from an input image.
    
    Args:
        img: PIL Image of a hand gesture
        
    Returns:
        str: Predicted letter (A-Y, excluding J and Z)
    """
    if img is None:
        return "Please provide an image"
    
    # Transform the image
    x = transform(img).unsqueeze(0)  # Shape: (1, 1, 28, 28)
    
    # Make prediction
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(1).item()
    
    # Map to letter
    predicted_letter = CLASS_TO_LETTER.get(pred_idx, str(pred_idx))
    
    return f"Predicted Letter: {predicted_letter}"


# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload or Draw Hand Gesture"),
    outputs=gr.Label(label="Prediction"),
    title="Sign Language MNIST CNN Classifier",
    description="""
    # Sign Language Letter Recognition
    
    Upload an image of a hand gesture representing an American Sign Language (ASL) letter.
    
    ## What images to use:
    - **Hand gestures** representing ASL letters A-Y (excluding J and Z)
    - Any image format (PNG, JPG, etc.)
    - Color or grayscale images work
    - Best results with clear, well-lit photos against simple backgrounds
    
    ## How it works:
    The model automatically:
    1. Converts your image to grayscale
    2. Resizes it to 28×28 pixels
    3. Predicts the corresponding letter
    
    **Note:** The model is trained on Sign Language MNIST dataset.
    For best results, use similar hand gesture images.
    """,
    examples=None,  # You can add example image paths here if available
    article="""
    ### About the Model
    This classifier uses a Convolutional Neural Network (CNN) trained on the Sign Language MNIST dataset.
    The dataset contains 24 classes (A-Y, excluding J and Z which require motion).
    
    ### Image Requirements
    - **Type**: Hand gesture photographs or drawings
    - **Format**: Any common format (PNG, JPG, JPEG, BMP, etc.)
    - **Processing**: Automatically converted to 28×28 grayscale
    - **Best practices**: 
      - Use clear, well-lit images
      - Simple background
      - Hand centered and clearly visible
      - Similar to American Sign Language hand positions
    """
)

if __name__ == "__main__":
    # Launch the interface
    demo.launch()
