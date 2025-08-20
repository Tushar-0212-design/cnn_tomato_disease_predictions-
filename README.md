 Tomato Disease Detection using CNN & MobileNetV2

<img width="1787" height="1017" alt="Screenshot 2025-08-08 071240" src="https://github.com/user-attachments/assets/1532ef8f-1872-4999-b705-a3d9fca1de37" />

 Overview

This project uses Deep Learning (CNN & Transfer Learning with MobileNetV2) to detect tomato plant leaf diseases from images.
It helps farmers, researchers, and agronomists by providing fast and accurate predictions via a simple Streamlit web app.

Objectives

Train a deep learning model to classify tomato leaf diseases.

Use data augmentation to improve model robustness.

Implement MobileNetV2 transfer learning for high accuracy.

Provide a real-time prediction interface.

Assist farmers with quick diagnosis to prevent crop loss.

 Dataset
images was taken from field .
Source: https://drive.google.com/drive/folders/1hdo_Ql48JfJ3p9Yi28piKmFRr_gzyogM?usp=drive_link

Classes:

Leaf Mold

Bacterial Spot

Late Blight

Yellow Leaf Curl Virus

Healthy

Mosaic Virus

Split:

Train → 70%

Validation → 15%

Test → 15%

📷 Preprocessing & Augmentation:

Resize: 150x150

Normalize: pixel values scaled 0–1

Augmentations: rotation, shift, shear, zoom, horizontal flip

🏗 Project Structure
cnn_tomato_disease_predictions/
│── tomato dataset.py         # Dataset splitting & renaming
│── tomato model.py           # Custom CNN training
│── TomatocnnV2.py            # MobileNetV2 training
│── st-tomatocnn.py           # Streamlit prediction app
│── tomato_disease_mobilenetv2_model.h5  # MobileNetV2 model
│── tomato_disease_cnn_model2.h5         # Custom CNN model
│── assets/                   # Images & plots for README
│── README.md                 # Documentation

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/Tushar-0212-design/cnn_tomato_disease_predictions-.git
cd cnn_tomato_disease_predictions-

2️⃣ Install Requirements
pip install tensorflow streamlit matplotlib seaborn scikit-learn pillow

3️⃣ Prepare Dataset

Place dataset in D:/Project/Tomato Augmantation (or update paths in code), then run:

python "tomato dataset.py"

4️⃣ Train Model

Option A – Custom CNN

python "tomato model.py"


Option B – MobileNetV2 (Recommended)

python "TomatocnnV2.py"

5️⃣ Launch Web App
streamlit run "st-tomatocnn.py"

📊 Model Performance
Model	Accuracy	Notes
MobileNetV2	~90%+	Faster, better generalization
Custom CNN	~85%+	Simpler, smaller

📈 Example Training Curves

<img width="386" height="278" alt="loss V2" src="https://github.com/user-attachments/assets/f70c6dee-0c2f-413a-81df-1de670a6a3e7" />
<img width="386" height="278" alt="accuracy V2" src="https://github.com/user-attachments/assets/8cd4d572-bceb-4e8c-b2b0-81c857617054" />




🌐 Web App Preview

<img width="1919" height="1016" alt="Screenshot 2025-08-08 071135" src="https://github.com/user-attachments/assets/08bb3da4-e066-4d44-87bf-0c79a26e73e1" />
