# TDSE: Exploring Convolutional Layers Through Data and Experiments

## Sign Language MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) for classifying American Sign Language (ASL) hand gestures using the Sign Language MNIST dataset.

## Features

- CNN model for sign language letter recognition
- Interactive Gradio web interface for predictions
- Supports letters A-Y (24 classes, excluding J and Z which require motion)
- Automatic image preprocessing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Gradio Interface

```bash
python app.py
```

This will launch a web interface where you can upload images of hand gestures for prediction.

## What Type of Images Can I Use? / ¿Qué tipo de imágenes podría utilizar?

### Supported Images:

1. **Content / Contenido**:
   - Hand gestures representing American Sign Language (ASL) letters
   - Gestos de manos representando letras del Lenguaje de Señas Americano (ASL)
   - Letters A-Y (excluding J and Z) / Letras A-Y (excluyendo J y Z)

2. **Format / Formato**:
   - Any common image format: PNG, JPG, JPEG, BMP, etc.
   - Cualquier formato de imagen común: PNG, JPG, JPEG, BMP, etc.

3. **Characteristics / Características**:
   - Color or grayscale (will be converted to grayscale automatically)
   - Color o escala de grises (se convertirá automáticamente a escala de grises)
   - Any size (will be resized to 28x28 pixels automatically)
   - Cualquier tamaño (se redimensionará automáticamente a 28x28 píxeles)

4. **Best Results / Mejores Resultados**:
   - Clear, well-lit photographs of hand gestures
   - Fotografías claras y bien iluminadas de gestos de manos
   - Simple backgrounds (preferably solid color)
   - Fondos simples (preferiblemente de color sólido)
   - Hand centered and clearly visible in the image
   - Mano centrada y claramente visible en la imagen
   - Similar to ASL alphabet hand positions
   - Similar a las posiciones de manos del alfabeto ASL

### Example Image Types:

✅ **Good:**
- Photos of hands making ASL letters
- Drawings or sketches of hand gestures
- Clear silhouettes of hand positions
- Images from the Sign Language MNIST dataset

❌ **Not Recommended:**
- Images with cluttered backgrounds
- Blurry or poorly lit photos
- Hand gestures not representing ASL letters
- Multiple hands in the same image

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers (32 and 64 filters)
- MaxPooling layers
- 2 Fully connected layers
- Dropout for regularization
- Output layer with 24 classes (letters A-Y excluding J and Z)

## Project Structure

```
.
├── app.py              # Gradio web interface
├── model.py            # CNN model definition
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dataset Information

This project is designed for the Sign Language MNIST dataset, which contains:
- 24 classes (letters A-Y, excluding J and Z)
- 28x28 grayscale images
- American Sign Language hand gestures

## License

See repository license for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.