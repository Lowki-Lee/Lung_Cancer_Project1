# Lung Cancer Risk Prediction System
Precented by Minhao Li & Zhonghao Guo
# Project Overview

This project aims to address the critical real-world problem of early lung cancer risk assessment. Lung cancer is one of the leading causes of cancer-related deaths globally. By utilizing machine learning techniques (Logistic Regression) on a dataset of physiological and lifestyle attributes, this tool allows users to:

1. Analyze historical patient data.

2. Train a predictive model to identify key risk factors.

3. Assess the risk probability for new patients through an interactive interface.

The solution is implemented in Python, adhering to Object-Oriented Programming (OOP) principles and modular software design.

# Features

Data Processing Pipeline: Automated data loading, cleaning, and feature encoding using the PatientData class.

Machine Learning Model: Uses Logistic Regression (via Scikit-learn) to predict malignancy probability.

Interactive Assessment: A robust user input module (PatientInput) that validates user responses and allows real-time risk prediction.

Visualization: Generates confusion matrices to visualize model performance.

Unit Testing: Includes Pytest cases to ensure data integrity and processing logic.

# Project Structure

The project is organized into modules for better maintainability:

Lung_Cancer_Project/
│
├── data/
│   ├── survey lung cancer.csv    # Kaggle Dataset (Public Domain)
│   └── test_data.csv             # Sample test data for prediction
│
├── src/                          # Source Code Modules
│   ├── patient_data.py           # Class: Handles data I/O and cleaning
│   ├── cancer_predictor.py       # Class: Handles model training and prediction
│   └── patient_input.py          # Class: Handles user interaction & validation
│
├── tests/                        # Unit Tests
│   └── test_project.py           # Pytest cases for data logic
│
├── main.ipynb                    # Main entry point (Jupyter Notebook)
├── README.md                     # Project documentation
└── requirements.txt              # List of dependencies

# Usage

The main program is encapsulated in a Jupyter Notebook for easy visualization and interaction.

1. Open the Notebook:
  Launch Jupyter Notebook or VS Code and open main.ipynb.

2. Run the Cells:
  Execute the cells in order. The notebook will:

  Initialize the CancerPredictor and PatientData classes.

  Clean and split the dataset.

  Train the Logistic Regression model.

  Display the accuracy score and Confusion Matrix.

3. Interactive Prediction:
  At the end of the notebook, an interactive loop will start. You will be prompted to enter patient details (e.g., Age, Smoking history, Anxiety levels).

  The PatientInput module will validate your inputs (ensuring they are numbers/within range).

  The system will output the Probability of Lung Cancer based on the trained model.

# Testing

This project uses pytest to ensure the robustness of data processing. To run the tests:

1. Open your terminal in the project root directory.

2. Run the following command:

python -m pytest


3. You should see a success message (e.g., 2 passed) indicating the data loading and cleaning logic is correct.

# Libraries Used 

Pandas: For data manipulation and CSV I/O.

Scikit-learn: For Logistic Regression model and metrics.

Matplotlib & Seaborn: For data visualization (Heatmaps).

OS & Sys: For robust file path management.

Pytest: For unit testing.

# Contributors 

[Minhao Li]: Project Structure, Model Implementation, Data Logic. 
Email: mli105@stevens.edu

[Zhonghao Guo]: Input Validation Module (patient_input.py), User Interaction flow.
Email: guozhonghao2002@gmail.com

# This project was developed for the AAI/CPE/EE 551 course.