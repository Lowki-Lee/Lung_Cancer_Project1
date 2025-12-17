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