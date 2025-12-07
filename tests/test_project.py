from pathlib import Path
import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patient_data import PatientData

@pytest.fixture
def mock_csv(tmp_path: Path):
    d = {
        'GENDER': ['M', 'F', 'M'],
        'AGE': [60, 50, 70],
        'LUNG_CANCER': ['YES', 'NO', 'YES'],
        'SMOKING': [2, 1, 2],
        'ANXIETY': [2, 1, 2],
        'YELLOW_FINGERS': [2, 1, 2]
    }

    for col in ['PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 
                'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                'SWALLOWING DIFFICULTY', 'CHEST PAIN']:
        d[col] = [1, 1, 1]

    df = pd.DataFrame(d)
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_load_data_success(mock_csv: str):
    """Test 1: Can the data be loaded successfully?"""
    pd_obj = PatientData(mock_csv)
    assert pd_obj.data is not None
    assert len(pd_obj.data) == 3

def test_clean_data_logic(mock_csv: str):
    """Test 2: Is the cleaning logic correct? (YES->1, M->1)"""
    pd_obj = PatientData(mock_csv)
    pd_obj.clean_data()

    assert pd_obj.data.iloc[0]['GENDER'] == 1

    assert pd_obj.data.iloc[0]['LUNG_CANCER'] == 1
#To avoid "module not found" path errors, please run the following command in the project root directory:
#python -m pytest
