from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import tqdm
from lightgbm import LGBMModel, log_evaluation
from lightgbm.callback import CallbackEnv
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from typing_extensions import Self

from lightgbm_callbacks import DartEarlyStoppingCallback

from .._tqdm_callback import ProgressBarCallback
from ._base import EstimatorWrapperBase
from ._dart_early_stopping_wrapper import LGBMDartEarlyStoppingSimpleWrapper


class LGBMEarlyStoppingEstimator(EstimatorWrapperBase[LGBMModel]):
    """LightGBM wrapper that does early stopping with sklearn.train_test_split."""

    def __init__(
        self,
        estimator: LGBMModel,
        *,
        early_stopping_factory: Callable[
            [int, bool, bool, float | list[float]], Callable[[CallbackEnv], None]
        ]
        | Callable[
            [int, bool, bool], Callable[[CallbackEnv], None]
        ] = DartEarlyStoppingCallback,
        stopping_rounds: int | None = None,
        first_metric_only: bool = False,
        verbose: bool = False,
        min_delta: float | None = None,
        eval_metric: str
        | Callable[[NDArray, NDArray], tuple[str, float, bool]]
        | None = None,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int = 0,
        shuffle: bool = True,
        stratify: bool = False,
        split_enabled: bool = True,
        tqdm_cls: Literal[
            "auto",
            "autonotebook",
            "std",
            "notebook",
            "asyncio",
            "keras",
            "dask",
            "tk",
            "gui",
            "rich",
            "contrib.slack",
            "contrib.discord",
            "contrib.telegram",
            "contrib.bells",
        ]
        | type[tqdm.std.tqdm]
        | None = None,
        tqdm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """LightGBM wrapper that does early stopping with sklearn.train_test_split.

        Parameters
        ----------
        estimator : LGBMModel
            Scikit-learn API LightGBM estimator.
        stopping_rounds : int
            The possible number of rounds without the trend occurrence.
            Alias: ``n_iter_no_change``
        early_stopping_factory : Callable[[int, bool, bool, float | list[float]],
        Callable[CallbackEnv, None]] | Callable[[int, bool, bool], Callable[CallbackEnv, None]],
        optional
            Factory function that returns a callback function, by default DartEarlyStoppingCallback
        first_metric_only : bool, optional
            Whether to use only the first metric for early stopping or use all of them, by default False
        min_delta : float, optional
            Minimum delta to be considered as an actual change, by default 0.0
            Alias: ``tol``
        verbose : bool, optional
            Whether to print message about early stopping, by default False
        eval_metric : str | Callable[[NDArray, NDArray], tuple[str, float, bool]], optional
            Evaluation metric, by default "rmse"
        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.
            Alias: ``validation_fraction``
        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
            Alias: ``train_fraction``
        random_state : int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        shuffle : bool, optional
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None. by default True
        stratify : bool, optional
            Whether or not stratify the data before splitting. If stratify=True,
            y must be categorical. by default False
        split_enabled : bool, optional
            Whether to use train_test_split or not, by default True
        tqdm_cls : Literal['auto', 'autonotebook', 'std', 'notebook', 'asyncio',
        'keras', 'dask', 'tk', 'gui', 'rich', 'contrib.slack', 'contrib.discord',
        'contrib.telegram', 'contrib.bells'] or type[tqdm.std.tqdm] or None, optional
            The tqdm class or module name, by default 'auto'
        tqdm_kwargs : dict[str, Any] or None, optional
            The keyword arguments passed to the tqdm class initializer, by default None
        **kwargs : Any
            Other parameters passed to the estimator.
        """
        self.estimator = estimator
        self.early_stopping_factory = early_stopping_factory
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta
        self.eval_metric = eval_metric
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.split_enabled = split_enabled
        self.tqdm_cls = tqdm_cls
        self.tqdm_kwargs = tqdm_kwargs
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if key not in [
                "validation_fraction",
                "train_fraction",
                "tol",
                "n_iter_no_change",
            ]:
                warnings.warn(f"Unknown parameter: {key}: {value}")

    def fit(self, X, y=None, **fit_params) -> Self:  # type: ignore
        """Fit the model according to the given training data."""
        if not self.split_enabled:
            self.estimator.fit(X, y, **fit_params)
            return self
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size or self.kwargs.get("validation_fraction", None),
            train_size=self.train_size or self.kwargs.get("train_fraction", None),
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=y if self.stratify else None,
        )
        fit_params["eval_set"] = [(X_train, y_train), (X_test, y_test)]
        if self.eval_metric is not None:
            fit_params["eval_metric"] = self.eval_metric
        stopping_rounds = self.stopping_rounds or self.kwargs.get(
            "n_iter_no_change", None
        )
        if stopping_rounds is not None:
            try:
                early_stopping = self.early_stopping_factory(
                    stopping_rounds,
                    self.first_metric_only,
                    self.verbose,
                    self.min_delta or self.kwargs.get("tol", 0.0),  # type: ignore
                )
            except TypeError:
                early_stopping = self.early_stopping_factory(
                    stopping_rounds, self.first_metric_only, self.verbose  # type: ignore
                )
            fit_params["callbacks"] = [
                early_stopping,
                log_evaluation(self.verbose),
            ] + fit_params.get("callbacks", [])
            for callback in fit_params["callbacks"]:
                if isinstance(callback, ProgressBarCallback):
                    callback.early_stopping_callback = early_stopping
            if self.tqdm_cls is not None:
                fit_params["callbacks"].append(
                    ProgressBarCallback(
                        tqdm_cls=self.tqdm_cls,
                        early_stopping_callback=early_stopping,
                        **(self.tqdm_kwargs or {}),
                    )
                )
        self.estimator.fit(
            X_train,
            y_train,
            **fit_params,
        )
        return self


class LGBMDartEarlyStoppingEstimator(LGBMEarlyStoppingEstimator):
    """LightGBM wrapper that does early stopping with sklearn.train_test_split
    and uses ``LGBMDartEarlyStoppingSimpleWrapper`` to support early stopping in dart mode.
    """

    def __init__(
        self,
        estimator: LGBMModel,
        *,
        early_stopping_factory: Callable[
            [int, bool, bool, float | list[float]], Callable[[CallbackEnv], None]
        ]
        | Callable[
            [int, bool, bool], Callable[[CallbackEnv], None]
        ] = DartEarlyStoppingCallback,
        stopping_rounds: int | None = None,
        first_metric_only: bool = False,
        verbose: bool = False,
        min_delta: float | None = None,
        eval_metric: str
        | Callable[[NDArray, NDArray], tuple[str, float, bool]]
        | None = None,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: int = 0,
        shuffle: bool = True,
        stratify: bool = False,
        split_enabled: bool = True,
        dart_early_stopping_method: Literal["save", "refit", "none"] = "save",
        metric_idx: int = -1,
        tqdm_cls: Literal[
            "auto",
            "autonotebook",
            "std",
            "notebook",
            "asyncio",
            "keras",
            "dask",
            "tk",
            "gui",
            "rich",
            "contrib.slack",
            "contrib.discord",
            "contrib.telegram",
            "contrib.bells",
        ]
        | type[tqdm.std.tqdm]
        | None = None,
        tqdm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """LightGBM wrapper that does early stopping with sklearn.train_test_split
        and uses ``LGBMDartEarlyStoppingSimpleWrapper`` to support early stopping in dart mode.

        Parameters
        ----------
        estimator : LGBMModel
            Scikit-learn API LightGBM estimator.
        stopping_rounds : int
            The possible number of rounds without the trend occurrence.
            Alias: ``n_iter_no_change``
        early_stopping_factory : Callable[[int, bool, bool, float | list[float]],
        Callable[CallbackEnv, None]] | Callable[[int, bool, bool], Callable[CallbackEnv, None]],
        optional
            Factory function that returns a callback function, by default DartEarlyStoppingCallback
        first_metric_only : bool, optional
            Whether to use only the first metric for early stopping or use all of them, by default False
        min_delta : float, optional
            Minimum delta to be considered as an actual change, by default 0.0
            Alias: ``tol``
        verbose : bool, optional
            Whether to print message about early stopping, by default False
        eval_metric : str | Callable[[NDArray, NDArray], tuple[str, float, bool]], optional
            Evaluation metric, by default "rmse"
        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.
            Alias: ``validation_fraction``
        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
            Alias: ``train_fraction``
        random_state : int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        shuffle : bool, optional
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None. by default True
        stratify : bool, optional
            Whether or not stratify the data before splitting. If stratify=True,
            y must be categorical. by default False
        split_enabled : bool, optional
            Whether to use train_test_split or not, by default True
        dart_early_stopping_method : Literal["save", "refit", "none"], optional
            Method to use for early stopping, by default "save"
        metric_idx : int, optional
            Index of the metric to use for early stopping, by default 0
        tqdm_cls : Literal['auto', 'autonotebook', 'std', 'notebook', 'asyncio',
        'keras', 'dask', 'tk', 'gui', 'rich', 'contrib.slack', 'contrib.discord',
        'contrib.telegram', 'contrib.bells'] or type[tqdm.std.tqdm] or None, optional
            The tqdm class or module name, by default 'auto'
        tqdm_kwargs : dict[str, Any] or None, optional
            The keyword arguments passed to the tqdm class initializer, by default None
        **kwargs : Any
            Other parameters passed to the estimator.
        """
        super().__init__(
            estimator,
            early_stopping_factory=early_stopping_factory,
            stopping_rounds=stopping_rounds,
            first_metric_only=first_metric_only,
            verbose=verbose,
            min_delta=min_delta,
            eval_metric=eval_metric,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
            split_enabled=split_enabled,
            tqdm_cls=tqdm_cls,
            tqdm_kwargs=tqdm_kwargs,
            **kwargs,
        )
        self.dart_early_stopping_method = dart_early_stopping_method
        self.metric_idx = metric_idx

    def fit(self, X, y=None, **fit_params) -> Self:  # type: ignore
        self.estimator = LGBMDartEarlyStoppingSimpleWrapper(
            self.estimator,
            method=self.dart_early_stopping_method,
            metric_idx=self.metric_idx,
        )
        super().fit(X, y, **fit_params)
        self.estimator = self.estimator.estimator
        return self
