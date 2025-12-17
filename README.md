# Lung Cancer Project

## Project Overview

This project is a **Python-based lung cancer risk prediction system** using machine learning. It is designed for coursework and learning purposes, demonstrating the complete workflow from data preprocessing to model training, evaluation, and interactive prediction for new patients.

The project covers key concepts including:

* Object-Oriented Programming (OOP)
* Data cleaning and preprocessing
* Machine learning model training and evaluation
* Exception handling
* Interactive user input
* Data visualization with Matplotlib and Seaborn



## Project Structure


Lung_Cancer_Project1/
│
├── data/
│   ├── survey lung cancer.csv   # Original lung cancer survey dataset (training data)
│   └── test_data.csv             # Sample test data for prediction
│
├── src/
│   ├── patient_data.py           # Data loading and preprocessing module
│   ├── cancer_predictor.py       # Model training, evaluation, and prediction
│   └── patient_input.py          # Interactive input for new patient prediction
│
├── tests/
│   └── test_project.py           # Unit tests (if applicable)
│
├── main.ipynb                    # Main entry point (recommended)
└── README.md                     # Project documentation
```



## File Descriptions

**1. patient_data.py**

This module is responsible for managing and preprocessing the dataset:

* Loads data from CSV files with exception handling
* Removes duplicate records
* Encodes categorical values into numerical format

  * GENDER: M / F → 1 / 0
  * YES / NO or 2 / 1 → 1 / 0
* Separates features (X) and target variable (y)

This module provides clean and structured data for model training.


**2. cancer_predictor.py**

This module handles model-related tasks:

* Trains a Logistic Regression model
* Splits data into training and testing sets
* Evaluates model performance using accuracy and classification report
* Visualizes results using a confusion matrix (Matplotlib & Seaborn)
* Provides a `predict_new_patient()` method for risk prediction

This is the core machine learning component of the project.



**3. patient_input.py**

This module enables user interaction for predicting lung cancer risk:

* Supports two input methods:

  * Read patient data from a CSV file (first row only)
  * Manually input patient data via command line
* Automatically infers required feature names
* Extracts probability values from different prediction output formats
* Displays:

  * Probability of lung cancer (percentage)
  * Simple risk recommendation (High / Low risk)

This module makes the model usable for end users.



**4. main.ipynb**

**Function: Main entry point (recommended way to run the project)**

* Initializes the cancer predictor
* Performs data preprocessing
* Trains the machine learning model
* Evaluates model performance
* Runs interactive patient risk prediction

 It is recommended that instructors or graders run this notebook directly.



## How to Run the Project

### Option 1: Run with Jupyter Notebook (Recommended)

1. Open the project in VS Code or Jupyter Notebook
2. Open `main.ipynb.`
3. Run all cells sequentially
4. Follow the prompts to input patient data and view predictions



### Option 2: Run via Command Line

```bash
python src/patient_input.py
```

When using CSV input, ensure the file path is correct, for example:

```
data/test_data.csv
```

---

## Input Features (Order Matters)

The model uses **15 features** in the following order:

1. GENDER
2. AGE
3. SMOKING
4. YELLOW_FINGERS
5. ANXIETY
6. PEER_PRESSURE
7. CHRONIC DISEASE
8. FATIGUE
9. ALLERGY
10. WHEEZING
11. ALCOHOL CONSUMING
12. COUGHING
13. SHORTNESS OF BREATH
14. SWALLOWING DIFFICULTY
15. CHEST PAIN

All features are converted into binary or numerical values before prediction.



## Example Output

Prediction result:
  Probability of disease: 73.42%
  Simple risk recommendation: High risk
```



## Technology Stack

* Python 3.x
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* Jupyter Notebook



## Notes

* Incorrect CSV file paths may cause `FileNotFoundError.`
* New patient input must contain exactly 15 features
* This project is for educational purposes only and does not provide medical advice



## Author

This project was developed as a coursework and learning project to demonstrate Python programming, data processing, and basic machine learning techniques.
