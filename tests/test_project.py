import sys
import os

# --- 新增这段代码 ---
# 获取当前文件的目录 (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (Lung_Cancer_Project)
parent_dir = os.path.dirname(current_dir)
# 将父目录加入到系统路径中，这样Python就能找到 'src' 包了
sys.path.append(parent_dir)
# ------------------

import pytest
import pandas as pd
from src.patient_data import PatientData
import os

# 创建一个临时的CSV文件用于测试
@pytest.fixture
def mock_csv(tmp_path):
    d = {
        'GENDER': ['M', 'F', 'M'],
        'AGE': [60, 50, 70],
        'LUNG_CANCER': ['YES', 'NO', 'YES'],
        'SMOKING': [2, 1, 2] # 2=Yes, 1=No
        # ... 可以添加更多列
    }
    df = pd.DataFrame(d)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_load_data_success(mock_csv):
    """测试数据加载成功"""
    pd_obj = PatientData(mock_csv)
    assert pd_obj.data is not None
    assert len(pd_obj.data) == 3

def test_clean_data_logic(mock_csv):
    """测试清洗逻辑是否正确转换了数值"""
    pd_obj = PatientData(mock_csv)
    pd_obj.clean_data()
    
    # 检查 M 是否变成了 1
    assert pd_obj.data.iloc[0]['GENDER'] == 1
    # 检查 YES 是否变成了 1
    assert pd_obj.data.iloc[0]['LUNG_CANCER'] == 1
    # 检查 SMOKING: 2->1, 1->0
    assert pd_obj.data.iloc[0]['SMOKING'] == 1
    assert pd_obj.data.iloc[1]['SMOKING'] == 0

def test_file_not_found():
    """测试文件不存在的异常捕捉"""
    with pytest.raises(FileNotFoundError):
        PatientData("non_existent_file.csv")