from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .model import Model
from apelid.utils.logging import get_logger

logger = get_logger(__name__)


class RandomForestModel(Model):
    """Random Forest classifier wrapper implementing the Model interface."""

    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.num_class = int(num_class)
        
        # Default hyperparameters optimized for multi-class classification
        base_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': int(random_state),
            'verbose': 0,
        }
        
        if params:
            base_params.update(params)
        
        self.model = RandomForestClassifier(**base_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info(f"[+] Random Forest training completed with {self.model.n_estimators} estimators")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    def save_model(self, path: str) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"[+] Random Forest model saved to {path}")

    @classmethod
    def load_model(cls, path: str, num_class: int = None) -> "RandomForestModel":
        import pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        inst = cls(num_class=num_class or 2)
        inst.model = model
        inst._is_fitted = True
        return inst
