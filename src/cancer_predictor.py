import sys
import os

import pandas as pd

# Get the directory of the current file (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Lung_Cancer_Project)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the system path so that Python can locate the 'src' package.
sys.path.append(parent_dir)
# ------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.patient_data import PatientData  # Import custom module

class CancerPredictor:
    """
    This class is responsible for model training, evaluation, and prediction.
Relationship: Composition (possesses an instance of PatientData)
    """
    def __init__(self, data_path):
        # Composition: Predictor 拥有 Data
        self.patient_data = PatientData(data_path)
        self.model = LogisticRegression(max_iter=1000)
        self.X_test = None
        self.y_test = None

    def run_preprocessing(self):
        """Invoke PatientData for cleansing"""
        self.patient_data.clean_data()

    def train_model(self):
        """Training the model"""
        X, y = self.patient_data.get_features_and_target()
        
        # Partitioning the training and test sets
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        print("Model training completed")

    def evaluate_model(self):
        """Evaluate the model and present the results"""
        if self.X_test is None:
            print("Please train the model first.")
            return

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # [Part 1] Requirement: Use Matplotlib/Seaborn
        self._plot_confusion_matrix(self.y_test, y_pred)

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Internal method: Constructing the confusion matrix"""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def predict_new_patient(self, features):
        """
        Predicting the risk of new patients
        features: list of values [GENDER, AGE, SMOKING, ...]
        """
        # Simple exception handling to ensure input length is correct
        if len(features) != 15: # 15 characteristics
             # [Part 1] Requirement: Exception handling (Part 2)
            raise ValueError(f"The number of features is mismatched; 15 are needed, but {len(features)} are provided.")

        X_input = pd.DataFrame([features], columns=self.model.feature_names_in_)
        probability = self.model.predict_proba(X_input)[0][1]
        return probability
