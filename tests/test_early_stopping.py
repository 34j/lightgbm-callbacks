from __future__ import annotations

from unittest import TestCase

import tqdm
import tqdm.rich
from lightgbm import LGBMRegressor, early_stopping
from parameterized import parameterized_class
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from lightgbm_callbacks import DartEarlyStoppingCallback, EarlyStoppingCallback


@parameterized_class(
    ("metric"),
    [
        (["l2", "l1"]),
    ],
)
class TestEarlyStoppingCallback(TestCase):
    tqdm_cls: type[tqdm.std.tqdm]
    tqdm_kwargs: dict[str, str]
    n_estimators: int = 100
    metric: list[str] | None

    def setUp(self) -> None:
        """load diabetes dataset"""
        self.X, self.y = load_diabetes(return_X_y=True)

    def test_normal_both_early_stopping_preds(self) -> None:
        """test our early stopping callback is consistent with the original early stopping"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        gbm = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
        )
        early_stopping_callback = EarlyStoppingCallback(
            stopping_rounds=50, first_metric_only=True
        )
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping_callback],
        )
        y_pred = gbm.predict(X_test)

        original_early_stopping_callback = early_stopping(
            stopping_rounds=50, first_metric_only=True
        )
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[original_early_stopping_callback],
        )
        y_pred_original = gbm.predict(X_test)

        self.assertEqual(y_pred.tolist(), y_pred_original.tolist())

        gbm_non_early_stopping = LGBMRegressor(
            n_estimators=early_stopping_callback.best_iter[0] + 1,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
        )
        gbm_non_early_stopping.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        y_pred_non_early_stopping = gbm_non_early_stopping.predict(X_test)

        self.assertEqual(y_pred.tolist(), y_pred_non_early_stopping.tolist())

    def test_normal_early_stopping_dart_preds(self) -> None:
        """test that early stopping results in the same model as normal training
        with the same number of iterations"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        gbm = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
            boosting_type="dart",
        )
        early_stopping_callback = EarlyStoppingCallback(
            stopping_rounds=50, first_metric_only=True
        )
        gbm.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping_callback],
        )
        y_pred = gbm.predict(X_test)

        gbm_non_early_stopping = LGBMRegressor(
            n_estimators=gbm.best_iteration_,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
            boosting_type="dart",
        )
        gbm_non_early_stopping.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        y_pred_non_early_stopping = gbm_non_early_stopping.predict(X_test)

        self.assertEqual(y_pred.tolist(), y_pred_non_early_stopping.tolist())
        # this result is worse than normal early stopping
        # but better than no early stopping

    def test_dart_early_stopping_not_final_iteration_preds(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        gbm = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
            boosting_type="dart",
        )
        early_stopping_callback = DartEarlyStoppingCallback(stopping_rounds=50)
        gbm.fit(
            X_train,
            y_train,
            callbacks=[early_stopping_callback],
            eval_set=[(X_test, y_test)],
        )

        self.assertLess(early_stopping_callback.best_iter[0], self.n_estimators)
        self.assertEqual(
            len(early_stopping_callback.best_model),
            len(early_stopping_callback.best_iter),
        )
        self.assertEqual(
            len(early_stopping_callback.best_model),
            len(early_stopping_callback.best_score),
        )
        # self.assertEqual(len(early_stopping_callback.best_model), len(self.metric)) ???

        for best_model, best_iter in zip(
            early_stopping_callback.best_model, early_stopping_callback.best_iter
        ):
            gbm_non_early_stopping = LGBMRegressor(
                n_estimators=best_iter + 1 + 1,
                learning_rate=1e-1,
                metric=self.metric,
                verbose=-1,
                boosting_type="dart",
            )
            gbm_non_early_stopping.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
            )
            self.assertEqual(
                best_model.predict(X_test).tolist(),
                gbm_non_early_stopping.predict(X_test).tolist(),
            )

    def test_dart_early_stopping_final_iteration_preds(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        n_estimators = 10
        gbm = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=1e-1,
            metric=self.metric,
            verbose=-1,
            boosting_type="dart",
        )
        early_stopping_callback = DartEarlyStoppingCallback(stopping_rounds=50)
        gbm.fit(
            X_train,
            y_train,
            callbacks=[early_stopping_callback],
            eval_set=[(X_test, y_test)],
        )

        self.assertEqual(early_stopping_callback.best_iter[0] + 1, n_estimators)
        self.assertEqual(
            len(early_stopping_callback.best_model),
            len(early_stopping_callback.best_iter),
        )
        self.assertEqual(
            len(early_stopping_callback.best_model),
            len(early_stopping_callback.best_score),
        )
        # self.assertEqual(len(early_stopping_callback.best_model), len(self.metric)) ???

        for best_model, best_iter in zip(
            early_stopping_callback.best_model, early_stopping_callback.best_iter
        ):
            gbm_non_early_stopping = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=1e-1,
                metric=self.metric,
                verbose=-1,
                boosting_type="dart",
            )
            gbm_non_early_stopping.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
            )
            self.assertEqual(
                best_model.predict(X_test).tolist(),
                gbm_non_early_stopping.predict(X_test).tolist(),
            )
