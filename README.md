

https://github.com/user-attachments/assets/7f188d45-aab5-4186-accc-84b49c48a482

# AICTE-Internship-Potato-Project
# Description of Project

**Dataset Preparation**

The dataset consists of 1,500 RGB images categorized into three classes.
Train: 900 images
Validation: 300 images
Test: 300 images

**Model Training (Train_potato_disease.ipynb)**

A deep learning model (likely a CNN-based architecture) is trained on the dataset.
Data augmentation and preprocessing techniques are applied to improve model generalization.
The trained model is saved as "trained_plant_disease_model_1.keras".

**Model Testing (Test_potato_disease.ipynb)**

The trained model is evaluated on the test dataset.
Performance metrics (accuracy, precision, recall, confusion matrix) are analyzed.

**Deployment using Streamlit (main.py)**

A web application is built using Streamlit to allow users to upload images of potato leaves for classification.
The uploaded image is preprocessed and passed to the trained model for prediction.
The result (predicted disease category) is displayed to the user.

# How to execute the project
First make sure all the version of libraries installed are compatible with each other. I have mentioned the versions used in "requirements.txt". In order to run the project make sure you are in the correct directory of main.py and run the following command:

**streamlit run main.py**
