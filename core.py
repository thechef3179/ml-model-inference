# core.py
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

# Pydantic for API validation
from pydantic import BaseModel

# Frameworks (wrapped in try-except to prevent crash if missing)
try:
    import joblib
except ImportError: joblib = None

try:
    import torch
    import torch.nn.functional as F
except ImportError: torch = None

try:
    import tensorflow as tf
except ImportError: tf = None


# --- 1. Data Models ---
class PredictionRequest(BaseModel):
    token: str
    datapoint: Dict[str, List[float]]


# --- 2. The Wrapper Interface ---
class ModelWrapper(ABC):
    def __init__(self, model: Any, framework: str):
        self.model = model
        self.framework = framework

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: pass


class SklearnWrapper(ModelWrapper):
    def predict(self, X: np.ndarray) -> np.ndarray: return self.model.predict(X)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"): return self.model.predict_proba(X)
        raise AttributeError("Model does not support predict_proba")

class PyTorchWrapper(ModelWrapper):
    def __init__(self, model, device='cpu'):
        super().__init__(model, 'PyTorch')
        self.device = device
        self.model.to(self.device).eval()

    def predict(self, X: np.ndarray):
        with torch.no_grad():
            t = torch.from_numpy(X).float().to(self.device)
            return torch.argmax(self.model(t), dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray):
        with torch.no_grad():
            t = torch.from_numpy(X).float().to(self.device)
            return F.softmax(self.model(t), dim=1).cpu().numpy()

class TensorFlowWrapper(ModelWrapper):
    def predict(self, X: np.ndarray): return np.argmax(self.model.predict(X), axis=1)
    def predict_proba(self, X: np.ndarray): return self.model.predict(X)


# --- 3. The Loader Factory ---
class ModelLoader:
    @staticmethod
    def load(path: str) -> ModelWrapper:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.pkl', '.joblib']:
            if not joblib: raise ImportError("joblib not installed")
            return SklearnWrapper(joblib.load(path), 'Scikit-Learn')
        elif ext in ['.pt', '.pth'] and torch:
            return PyTorchWrapper(torch.load(path))
        elif ext in ['.h5', '.keras'] and tf:
            return TensorFlowWrapper(tf.keras.models.load_model(path), 'TensorFlow')
        raise ValueError(f"Unsupported extension or missing library for {ext}")


# --- 4. Global State (The Singleton) ---
class GlobalState:
    def __init__(self):
        self.active_model: Optional[ModelWrapper] = None
        self.registry: Dict[str, ModelWrapper] = {}

state = GlobalState()
