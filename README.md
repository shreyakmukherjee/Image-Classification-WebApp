# Image Classification Web App

This project demonstrates image classification using two models: a custom CNN for the CIFAR-10 dataset and a pre-trained MobileNetV2 model. The project also includes a Streamlit web interface where users can upload images to classify them using the selected model.

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
- Confidence Score: 93.45%
---
### Contributing
Feel free to fork this repository, open issues, or submit pull requests if you have ideas for improvement.

---

### License
This project is licensed under the MIT License. See the LICENSE file for details.
```bash

---

### **Additional Notes**

1. **Requirements File (`requirements.txt`)**:
   Add the required dependencies:
```
tensorflow==2.10.0 streamlit numpy matplotlib pillow
```bash

2. **Adding Example Image**:
If you have an example output (e.g., a snapshot of the Streamlit interface), save it as `example_image.png` and reference it in the `README.md`.

3. **Testing Before Uploading**:
Test your scripts locally to ensure they work as expected before uploading to GitHub.

With this setup and instructions, your project will be well-structured, easy to understand, and ready for others to use!
```
