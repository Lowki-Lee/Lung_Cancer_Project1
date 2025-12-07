import pandas as pd
import os

class PatientData:
    """
    该类负责管理患者数据集，包括加载、清洗和预处理。
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self._load_data()

    def _load_data(self):
        """读取CSV文件，包含异常处理"""
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
        """数据清洗和预处理"""
        if self.data is None:
            return

        # 1. 去除重复值 (参考Kaggle分析)
        initial_count = len(self.data)
        self.data.drop_duplicates(inplace=True)
        print(f"Duplicate data removed: {initial_count - len(self.data)} records")

        # 2. 标签编码 (Mapping)
        # 根据Kaggle分析: YES=2, NO=1. 我们通常将其转换为 1 和 0 以便于机器学习
        # GENDER: M, F -> 1, 0
        # LUNG_CANCER: YES, NO -> 1, 0
        
        self.data['GENDER'] = self.data['GENDER'].map({'M': 1, 'F': 0})
        self.data['LUNG_CANCER'] = self.data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        # 其他列是 2(Yes) 和 1(No)，我们将它们转换为 1 和 0
        # [Part 2] Requirement: List comprehension (用于筛选需要转换的列)
        columns_to_fix = [col for col in self.data.columns if col not in ['GENDER', 'AGE', 'LUNG_CANCER']]
        
        for col in columns_to_fix:
            self.data[col] = self.data[col].map({2: 1, 1: 0})

        print("Data cleaning completed.")

    def get_features_and_target(self):
        """分离特征(X)和目标(y)"""
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
        # 假设你在项目根目录下运行，需要根据实际路径调整
        pd_obj = PatientData("data/survey lung cancer.csv")
        pd_obj.clean_data()
        print(pd_obj)
    except Exception as e:
        print(e)
