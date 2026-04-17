Keratoconus Detection using Deep Learning
This project implements a Full-Stack Deep Learning Application designed to assist in the early detection of Keratoconus, a progressive eye disease where the cornea thins and bulges into a cone-like shape.

The system uses a React (Vite) frontend to provide a professional user interface and a Flask (Python) backend to serve a MobileNetV2 neural network that classifies corneal topography maps.

🌟 Key Features
Three-Class Classification: Categorizes scans into Normal, Suspect (Early Stage), or Keratoconus.

Real-time Analysis: Asynchronous communication between the React UI and the Python AI engine.

Transfer Learning: Utilizes MobileNetV2 pre-trained on ImageNet for high feature extraction accuracy.

Responsive UI: Dynamic status indicators (Green/Yellow/Red) based on AI confidence levels.

🏗️ System Architecture
The application is split into two distinct layers to separate the user interface from the heavy computational requirements of TensorFlow.

📂 Why Two Python Scripts?
To maintain a clean development workflow, the backend is divided into two specialized files:

app.py (The Trainer/Researcher):

Role: Model development and evaluation.

Function: Contains the training loops, data augmentation logic, and validation metrics. Use this script only when you need to retrain the .h5 model or test its accuracy on new datasets.

flask_app.py (The Production Server):

Role: Live API Deployment.

Function: Loads the saved keratoconus_model.h5 and creates a REST API endpoint (/predict). It stays active to receive images from the React frontend, processes them, and returns JSON results.

🛠️ Technical Stack
Frontend: React.js, Vite, Tailwind CSS (for styling).

Backend: Flask, Flask-CORS.

AI/ML: TensorFlow 2.x, Keras, NumPy, Pillow (Image Processing).

Model Architecture: MobileNetV2 (Transfer Learning).

🚀 Getting Started
1. Prerequisites
Python 3.12 (Recommended for TensorFlow compatibility).

Node.js & npm (For the React frontend).

2. Backend Setup (Flask)
PowerShell
# Clone the repository
git clone https://github.com/AkilaKasun/keratoconus-detection-model.git
cd keratoconus-detection-model

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install flask flask-cors tensorflow pillow numpy
3. Frontend Setup (React)
Open a new terminal window:

PowerShell
# Navigate to your frontend folder
npm install
npm run dev
4. Configuration
Create a .env file in your React root directory:

Plaintext
VITE_API_URL=http://localhost:5000/predict
📊 Dataset Reference
The model is trained using the Keratoconus Detection dataset from Kaggle. It includes Axial, Sagittal, and Pachymetry corneal maps.

Normal: Healthy corneal structure.

Suspect: Early clinical signs (the primary target for this screening tool).

Keratoconus: Advanced thinning and bulging.

📈 Current Results
Training Accuracy: ~83%

Validation Accuracy: ~69%

Note: Validation accuracy is currently being optimized through advanced data augmentation and hyperparameter tuning.

🔮 Roadmap
[ ] Data Augmentation: Expand the dataset to improve model generalization.

[ ] Cloud Deployment: Host the Flask API on AWS/Google Cloud and the React UI on Vercel.

[ ] Mobile Integration: Convert the model to TensorFlow Lite for a dedicated mobile screening app.

👤 Author
Akila Kasun Full-Stack Developer


Disclaimer: This tool is intended for educational and screening assistance purposes only and should not replace professional medical diagnosis by an ophthalmologist.