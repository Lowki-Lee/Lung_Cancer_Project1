from typing import Any, Dict, List, Optional
import sys
import os
import pandas as pd
import numpy as np


def _extract_probability(pred: Any) -> float:
    """Heuristically extract a single floating-point probability value between [0..1] from the model output."""
    if pred is None:
        raise ValueError("Prediction is None.")
    if isinstance(pred, dict):
        for key in ("probability", "prob", "proba", "probabilities", "probs"):
            if key in pred:
                val = pred[key]
                try:
                    return float(np.asarray(val).ravel()[-1])
                except Exception:
                    pass
        for v in pred.values():
            try:
                return float(np.asarray(v).ravel()[-1])
            except Exception:
                continue
        raise ValueError("Cannot extract probability from dict prediction.")
    try:
        if isinstance(pred, (float, int, np.floating, np.integer)):
            return float(pred)
    except Exception:
        pass
    arr = np.asarray(pred)
    if arr.size == 1:
        return float(arr.ravel()[0])
    if arr.ndim == 1:
        return float(arr[-1])
    if arr.ndim == 2:
        return float(arr[0, -1])
    raise ValueError(f"Unrecognized prediction format: {type(pred)}")

def _get_expected_features_from_predictor(predictor: Any) -> Optional[List[str]]:
    """Try to infer the expected feature names from the predictor or its internal model."""
    attrs = [
        "feature_names",
        "feature_names_in_",
        "feature_names_",
        "features",
        "expected_features",
        "required_features",
        "feature_columns",
        "columns",
        "feature_names_in",
    ]
    for attr in attrs:
        val = getattr(predictor, attr, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple, np.ndarray, set)):
            return list(val)
        if isinstance(val, pd.Index):
            return list(val.tolist())
    # try nested 'model' attribute (common pattern)
    nested = getattr(predictor, "model", None)
    if nested is not None and nested is not predictor:
        for attr in attrs:
            val = getattr(nested, attr, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple, np.ndarray, set)):
                return list(val)
            if isinstance(val, pd.Index):
                return list(val.tolist())
    return None

def _ask_manual_input(feature_names: List[str]) -> Dict[str, Any]:
    """Request each feature value from the user in an interactive manner and return a dictionary."""
    print("Please enter patient data for the following features (press Enter):")
    patient: Dict[str, Any] = {}
    for feat in feature_names:
        raw = input(f"  {feat}: ").strip()
        if raw == "":
            val = None
        else:
            if raw.lower() in {"y", "yes", "true", "t", "m", "male"}:
                val = 1
            elif raw.lower() in {"n", "no", "false", "f", "female"}:
                val = 0
            else:
                try:
                    if "." in raw:
                        val = float(raw)
                    else:
                        val = int(raw)
                except Exception:
                    val = raw
        patient[feat] = val
    return patient

def _read_patient_from_csv(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    first_row = df.iloc[0]
    patient: Dict[str, Any] = {}
    for i in first_row.index:
        if isinstance(first_row[i], (np.integer, int)):
            if i in {"GENDER", "AGE"}:
                patient[i] = int(first_row[i])
            else:
                if int(first_row[i]) == 1:
                    patient[i] = 0
                else:
                    patient[i] = 1
        else:
            if first_row[i].lower() in {"y", "yes", "true", "t", "m", "male"}:
                patient[i] = 1
            elif first_row[i].lower() in {"n", "no", "false", "f", "female"}:
                patient[i] = 0
            else:
                patient[i] = first_row[i]
    return patient

def run_patient_input(project: Any) -> Dict[str, Any]:
    """
    Interactive entry point: It can accept input in CSV format or manual input to build a single patient record.
Call project.predict_new_patient(...) and print the probability as well as simple risk advice.
This project must provide a callable function predict_new_patient(DataFrame) -> numerical/probability dictionary/array.
    """
    if not hasattr(project, "predict_new_patient"):
        raise AttributeError("The provided project must have a 'predict_new_patient' method.")

    print("Select input method:")
    print("  1) Read from CSV (first row only)")
    print("  2) Manual input (interactive)")
    choice = input("Enter 1 or 2, then press Enter: ").strip()
    if choice not in {"1", "2"}:
        print("Invalid choice, defaulting to 2 (manual input).")
        choice = "2"
    if choice == "1":
        path = input("Enter CSV file path: ").strip()
        patient_raw = _read_patient_from_csv(path)
    else:
        print("2")
        feature_names = _get_expected_features_from_predictor(project) or []
        if not feature_names:
            print(
                "Could not infer feature names from the project. Please provide comma-separated feature names,"
                " for example: age,sex,smoking_history"
            )
            raw = input("Feature names (comma-separated): ").strip()
            feature_names = [s.strip() for s in raw.split(",") if s.strip()]
            if not feature_names:
                raise ValueError("No feature names provided.")
        patient_raw = _ask_manual_input(feature_names)

    patient = []
    for i in patient_raw:
        patient.append(patient_raw[i])

    predict_fn = getattr(project, "predict_new_patient")
    if not callable(predict_fn):
        raise TypeError("project.predict_new_patient is not callable.")
    pred = predict_fn(patient)
    prob = _extract_probability(pred)

    prob_pct = prob * 100.0
    risk = "High risk" if prob >= 0.5 else "Low risk"
    print("\nPrediction result:")
    print(f"  Probability of disease: {prob_pct:.2f}%")
    print(f"  Simple risk suggestion: {risk}")

    return {"probability": prob, "risk": risk, "patient_raw": patient_raw}