Keratoconus Detection using Deep Learning
This project implements a Deep Learning model to assist in the early detection of Keratoconus, a progressive eye disease where the cornea thins and begins to bulge into a cone-like shape. By analyzing corneal topography maps, this AI model classifies images into three categories: Normal, Suspect, or Keratoconus.

📌 Project Overview
The goal of this project is to provide a screening tool that can identify early-stage Keratoconus (Suspect) which is often difficult to detect during routine eye exams. The model utilizes Transfer Learning with the MobileNetV2 architecture to achieve high accuracy with minimal computational resources.

📊 Dataset
The model is trained using the Keratoconus Detection dataset from Kaggle.

Dataset Link: Keratoconus Detection on Kaggle

Content: The dataset includes corneal topography maps (Axial, Sagittal, and Pachymetry) categorized into:

Normal: Healthy eyes.

Suspect: Eyes showing early clinical signs of thinning.

Keratoconus: Confirmed cases of the disease.

🛠️ Technical Stack
Language: Python

Environment: Google Colab (T4 GPU)

Framework: TensorFlow / Keras

Base Model: MobileNetV2 (Pre-trained on ImageNet)

Libraries: NumPy, Matplotlib, ImageDataGenerator

🚀 How to Run the Project
1. Prerequisites
You will need a Kaggle account and a kaggle.json API token to download the data directly into your environment.

2. Installation & Setup
Clone this repository and open the notebook in Google Colab:

Bash
git clone https://github.com/AkilaKasun/keratoconus-detection-model.git
3. Training the Model
The notebook follows these steps:

Data Extraction: Unzipping and organizing the Kaggle dataset.

Preprocessing: Rescaling images to 224x224 pixels and normalizing pixel values.

Model Architecture: * Frozen MobileNetV2 base.

Global Average Pooling layer.

Dense hidden layer (128 units, ReLU).

Dropout layer (20%) to prevent overfitting.

Softmax output layer (3 units).

Training: The model is trained for 10 epochs using the Adam optimizer.

📈 Results
The model currently achieves:

Training Accuracy: ~83%

Validation Accuracy: ~69% (Ongoing improvements for generalization)

🔮 Future Roadmap
Web Interface: Deploying the model as a web application using Streamlit or React.

Data Augmentation: Increasing the variety of training images to improve validation accuracy.

Mobile App: Converting the model to TensorFlow Lite for real-time mobile screening.