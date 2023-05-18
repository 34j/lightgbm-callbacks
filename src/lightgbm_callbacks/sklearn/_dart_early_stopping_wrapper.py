from __future__ import annotations

import warnings
from logging import getLogger
from typing import Any, Literal

from lightgbm import LGBMModel
from typing_extensions import Self

from .._early_stopping_callback import DartEarlyStoppingCallback, EarlyStoppingCallback
from ._base import EstimatorWrapperBase

LOG = getLogger(__name__)


class LGBMDartEarlyStoppingSimpleWrapper(EstimatorWrapperBase[LGBMModel]):
    """A simple wrapper for dart LGBMModel that returns the best model after early stopping."""

    def __init__(
        self,
        estimator: LGBMModel,
        *,
        method: Literal["save", "refit", "none"] = "save",
        metric_idx: int = -1,
    ) -> None:
        """A simple wrapper for dart LGBMModel that returns the best model after early stopping.

        Parameters
        ----------
        estimator : LGBMModel
            The estimator to be wrapped.
        method : Literal[&quot;save&quot;, &quot;refit&quot;, &quot;none&quot;], optional
            The method to return the best model, by default &quot;save&quot;
            "save": Save the best model by deepcopying the estimator and return the best model.
            "refit": Refit the estimator with the best iteration and return the refitted estimator.
            "none": Do nothing and return the original estimator.
        metric_idx : int, optional
            The index of the metric to be used for early stopping, by default 0
        """
        self.method = method
        self.metric_idx = metric_idx
        super().__init__(estimator)

    def fit(self, X, y, **fit_params: Any) -> Self:  # type: ignore
        early_stopping_callback_candidates = [
            callback
            for callback in fit_params.get("callbacks", [])
            if isinstance(callback, EarlyStoppingCallback)
        ]
        if (
            self.estimator.get_params()["boosting_type"] != "dart"
            or (len(early_stopping_callback_candidates) == 0)
            or (self.method == "none")
        ):
            self.estimator.fit(X, y, **fit_params)
            return self

        if len(early_stopping_callback_candidates) > 1:
            warnings.warn(
                "Multiple EarlyStoppingCallback objects are found. "
                "Only the first one is used.",
                UserWarning,
            )
        early_stopping_callback = early_stopping_callback_candidates[0]

        if self.method in ["refit", "refit_like_save"]:
            self.estimator.fit(X, y, **fit_params)
            LOG.debug(f"best_iter: {early_stopping_callback.best_iter}")
            self.estimator.set_params(
                n_estimators=early_stopping_callback.best_iter[self.metric_idx]
                + 1
                + (1 if self.method == "refit_like_save" else 0)
            )
            self.estimator.fit(X, y, **fit_params)

        elif self.method == "save":
            if not isinstance(early_stopping_callback, DartEarlyStoppingCallback):
                raise ValueError(
                    "EarlyStoppingCallback is not DartEarlyStoppingCallback. "
                    f"Got {type(early_stopping_callback)}"
                )
            self.estimator.fit(X, y, **fit_params)
            self.estimator._Booster = early_stopping_callback.best_model[
                self.metric_idx
            ]

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self
