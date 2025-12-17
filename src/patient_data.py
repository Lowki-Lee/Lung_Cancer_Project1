import pandas as pd
import os

class PatientData:
    """
    This class is responsible for managing patient datasets, including loading, cleaning, and preprocessing.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self._load_data()

    def _load_data(self):
        """Reading CSV files, including exception handling"""
        # [Part 2] Requirement: Built-in library (os)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"file not found: {self.file_path}")
        
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully, {len(self.data)} records.")
        except Exception as e:
            # [Part 1] Requirement: Exception handling
            raise ValueError(f"Error reading CSV file: {e}")

    def clean_data(self):
        """Data cleansing and preprocessing"""
        if self.data is None:
            return

        # 1. Remove duplicate values (referencing Kaggle analysis)
        initial_count = len(self.data)
        self.data.drop_duplicates(inplace=True)
        print(f"Duplicate data removed: {initial_count - len(self.data)} records")

        # 2. Tag Mapping
        # According to Kaggle analysis: YES=2, NO=1. We typically convert these to 1 and 0 for machine learning purposes.
        # GENDER: M, F -> 1, 0
        # LUNG_CANCER: YES, NO -> 1, 0
        
        self.data['GENDER'] = self.data['GENDER'].map({'M': 1, 'F': 0})
        self.data['LUNG_CANCER'] = self.data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        # The other columns are 2 (Yes) and 1 (No), which we convert to 1 and 0.
        # [Part 2] Requirement: List comprehension (Used to filter columns requiring conversion)
        columns_to_fix = [col for col in self.data.columns if col not in ['GENDER', 'AGE', 'LUNG_CANCER']]
        
        for col in columns_to_fix:
            self.data[col] = self.data[col].map({2: 1, 1: 0})

        print("Data cleaning completed.")

    def get_features_and_target(self):
        """Separating features (X) and target (y)"""
        X = self.data.drop('LUNG_CANCER', axis=1)
        y = self.data['LUNG_CANCER']
        return X, y

    def __str__(self):
        # [Part 2] Requirement: __str__
        if self.data is None:
            return "PatientData: Empty"
        return f"PatientData Object: {len(self.data)} records, Columns: {list(self.data.columns)}"

# [Part 2] Requirement: if __name__ == "__main__"
if __name__ == "__main__":
    # 测试
    try:
        # Assuming you are running from the project root directory, adjust the path accordingly based on your actual setup.
        pd_obj = PatientData("data/survey lung cancer.csv")
        pd_obj.clean_data()
        print(pd_obj)
    except Exception as e:
        print(e)
