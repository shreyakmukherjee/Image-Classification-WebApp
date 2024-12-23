# Image Classification Web App

This project demonstrates image classification using two models: a custom CNN for the CIFAR-10 dataset and a pre-trained MobileNetV2 model. The project also includes a Streamlit web interface where users can upload images to classify them using the selected model.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Example Usage](#example-usage)
5. [Example Output](#example-output)
    - [Streamlit Interface](#streamlit-interface)
    - [Sample Prediction](#sample-prediction)
6. [Contributing](#contributing)

## Project Structure

- `cifar10_cnn.py`: Standalone script for training and testing a CNN on the CIFAR-10 dataset.
- `mobilenet_v2.py`: Standalone script for using the MobileNetV2 model for classification.
- `app.py`: Streamlit web application to interact with both models and classify uploaded images.
- `requirements.txt`: Python dependencies for running the project.
- `README.md`: Description and instructions for the project.
- `example_image.png` (optional): Sample image for testing.


## Features

1. **Custom CNN**:
   - Trained on the CIFAR-10 dataset.
   - Can classify 10 categories such as airplanes, cars, cats, and dogs.

2. **MobileNetV2**:
   - Pre-trained on ImageNet.
   - Fine-tuned for CIFAR-10-like classification tasks.

3. **Web Interface**:
   - Built with Streamlit.
   - Allows users to upload an image and select the desired model for classification.
   - Displays the predicted class and confidence score.

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- Git installed on your system

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Image-Classification-WebApp.git
   cd Image-Classification-WebApp
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
4. Open the app in your browser (usually at http://localhost:8501).
---
### Example Usage
- Select a model from the sidebar (CIFAR-10 CNN or MobileNetV2).
- Upload an image for classification.
- View the predicted class and confidence score.
---

# Example Output
## Streamlit Interface
### Sample Prediction

- Uploaded Image: A cat.
- Predicted Class: Cat
- Confidence Score: 84.77%
---
### Contributing
Feel free to fork this repository, open issues, or submit pull requests if you have ideas for improvement.

