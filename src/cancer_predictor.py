import sys
import os


# 获取当前文件的目录 (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (Lung_Cancer_Project)
parent_dir = os.path.dirname(current_dir)
# 将父目录加入到系统路径中，这样Python就能找到 'src' 包了
sys.path.append(parent_dir)
# ------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.patient_data import PatientData  # 导入自定义模块

class CancerPredictor:
    """
    该类负责模型训练、评估和预测。
    关系: Composition (拥有一个 PatientData 实例)
    """
    def __init__(self, data_path):
        # Composition: Predictor 拥有 Data
        self.patient_data = PatientData(data_path)
        self.model = LogisticRegression(max_iter=1000)
        self.X_test = None
        self.y_test = None

    def run_preprocessing(self):
        """调用 PatientData 进行清洗"""
        self.patient_data.clean_data()

    def train_model(self):
        """训练模型"""
        X, y = self.patient_data.get_features_and_target()
        
        # 划分训练集和测试集
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        print("Model training completed")

    def evaluate_model(self):
        """评估模型并展示结果"""
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
        """内部方法：绘制混淆矩阵"""
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
        预测新患者的风险
        features: list of values [GENDER, AGE, SMOKING, ...]
        """
        # 简单的异常处理，确保输入长度正确
        if len(features) != 15: # 15个特征
             # [Part 1] Requirement: Exception handling (Part 2)
            raise ValueError(f"The number of features is mismatched; 15 are needed, but {len(features)} are provided.")
        
        probability = self.model.predict_proba([features])[0][1]
        return probability
