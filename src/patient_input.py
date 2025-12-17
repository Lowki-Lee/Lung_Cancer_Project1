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