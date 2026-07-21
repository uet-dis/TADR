from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import lightgbm as lgb
from .model import Model
from apelid.utils.logging import get_logger

logger = get_logger(__name__)


class LGBMModel(Model):
    """LightGBM classifier wrapper implementing the Model interface."""

    def __init__(
        self,
        *,
        num_class: int,
        params: Optional[Dict[str, Any]] = None,
        num_round: int = 200,
        early_stopping: int = 20,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.num_class = int(num_class)
        self.num_round = int(num_round)
        self.early_stopping = int(early_stopping)
        
        # Default hyperparameters optimized for multi-class classification
        self.params = {
            'objective': 'multiclass',
            'num_class': self.num_class,
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'random_state': int(random_state),
            'verbose': -1,
        }
        
        if params:
            self.params.update(params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        train_data = lgb.Dataset(X_train, label=y_train)
        
        eval_sets = None
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            eval_sets = [val_data]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_round,
            valid_sets=eval_sets,
            callbacks=[lgb.early_stopping(self.early_stopping)] if eval_sets else [],
        )
        self._is_fitted = True
        logger.info(f"[+] LightGBM training completed. Best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def save_model(self, path: str) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.save_model(path)
        logger.info(f"[+] LightGBM model saved to {path}")

    @classmethod
    def load_model(cls, path: str, num_class: int = None) -> "LGBMModel":
        model = lgb.Booster(model_file=path)
        inst = cls(num_class=num_class or 2)
        inst.model = model
        inst._is_fitted = True
        return inst
