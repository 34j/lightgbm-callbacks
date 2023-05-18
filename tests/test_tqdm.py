from unittest import TestCase

import tqdm
import tqdm.rich
from lightgbm import LGBMRegressor
from parameterized import parameterized_class
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from lightgbm_callbacks import EarlyStoppingCallback, ProgressBarCallback


@parameterized_class(
    ("tqdm_cls", "tqdm_kwargs", "metric"),
    [
        ("auto", {}, None),
        (tqdm.rich.tqdm, {"desc": "Rich"}, ["rmse", "l2"]),
    ],
)
class TestProgressBarCallback(TestCase):
    tqdm_cls: type[tqdm.std.tqdm]
    tqdm_kwargs: dict[str, str]
    n_estimators: int = 100
    metric: list[str] | None

    def setUp(self) -> None:
        self.X, self.y = load_diabetes(return_X_y=True)

    def test_fit(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        clf = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=1e-1,
            random_state=0,
            metric=self.metric,
            verbose=-1,
        )
        callback = ProgressBarCallback(self.tqdm_cls, **self.tqdm_kwargs)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[callback],
        )

        self.assertTrue(issubclass(callback.tqdm_cls, tqdm.std.tqdm))
        self.assertIsInstance(callback.pbar, tqdm.std.tqdm)
        assert callback.pbar is not None
        self.assertEqual(callback.pbar.total, self.n_estimators)
        self.assertEqual(callback.pbar.n, self.n_estimators)

    def test_early_stopping(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        clf = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
        )
        early_stopping_callback = EarlyStoppingCallback(stopping_rounds=50)
        callback = ProgressBarCallback(
            self.tqdm_cls,
            early_stopping_callback=early_stopping_callback,
            **self.tqdm_kwargs
        )
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping_callback, callback],
        )

    def test_warn_override(self) -> None:
        with self.assertWarns(UserWarning):
            ProgressBarCallback(self.tqdm_cls, total=100, **self.tqdm_kwargs)
