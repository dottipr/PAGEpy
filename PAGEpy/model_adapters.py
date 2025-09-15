import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import roc_auc_score

from PAGEpy.models import NNModelConfig

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    def train_and_score(self,
                        x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray,
                        run_params: dict
                        ) -> float:
        """Train model and return AUC score."""


class NeuralNetworkAdapter(ModelAdapter):
    """Adapter for neural network models."""

    def __init__(self, model_class, model_params, n_features: int):
        self.model_class = model_class
        self.model_params = model_params
        self.n_features = n_features

    def train_and_score(self, x_train, y_train, x_test, y_test, run_params: dict) -> float:
        """Train NN model and return AUC score."""
        try:
            model = self.model_class(
                n_input_features=self.n_features,
                config=NNModelConfig(**self.model_params)
            )
            model.train(x_train=x_train, y_train=y_train, **run_params)
            return model.evaluate(x_test, y_test)
        except Exception as e:
            logger.error("Neural network training failed: %s", str(e))
            return 0.0


class SklearnAdapter(ModelAdapter):
    """Adapter for scikit-learn style models."""

    def __init__(self, model_class, model_params):
        self.model_class = model_class
        self.model_params = model_params

    def train_and_score(self, x_train, y_train, x_test, y_test, run_params: dict) -> float:
        """Train sklearn model and return AUC score."""
        try:
            model = self.model_class(**self.model_params)
            model.fit(x_train, y_train, **run_params)

            # Get AUC score based on available prediction methods
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(x_test)[:, 1]
                return float(roc_auc_score(y_test, y_pred_proba))
            elif hasattr(model, 'decision_function'):
                y_pred_scores = model.decision_function(x_test)
                return float(roc_auc_score(y_test, y_pred_scores))
            else:
                # Fallback to accuracy
                y_pred = model.predict(x_test)
                return np.mean(y_pred == y_test)

        except Exception as e:
            logger.error("Sklearn model training failed: %s", str(e))
            return 0.0


class ModelAdapterFactory:
    """
    Factory for creating appropriate model adapters.
    Update this class as well when implementing new models.
    """

    @staticmethod
    def create_adapter(model_class, model_params, n_features: int) -> ModelAdapter:
        """Create appropriate model adapter based on model type."""
        if hasattr(model_class, '__name__') and 'NN' in model_class.__name__:
            return NeuralNetworkAdapter(model_class, model_params, n_features)
        else:
            return SklearnAdapter(model_class, model_params)
