from __future__ import annotations

from unittest import TestCase

import lightgbm as lgb
import matplotlib.pyplot as plt
import tqdm
import tqdm.rich
from lightgbm import LGBMRegressor
from parameterized import parameterized_class
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lightgbm_callbacks import LGBMDartEarlyStoppingEstimator, ProgressBarCallback


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
        self.X, self.y = load_diabetes(return_X_y=True)

    def test_performance(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, random_state=0
        )
        gbms = {
            "baseline": LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=1e-1,
                metric=self.metric,
                verbose=-1,
                boosting_type="dart",
            )
        }
        gbms.update(
            {
                method: LGBMDartEarlyStoppingEstimator(
                    LGBMRegressor(
                        n_estimators=self.n_estimators,
                        learning_rate=1e-1,
                        metric=self.metric,
                        verbose=-1,
                        boosting_type="dart",
                    ),
                    stopping_rounds=30,
                    dart_early_stopping_method=method,  # type: ignore
                    random_state=0,
                    metric_idx=-1,
                )
                for method in ["refit", "refit_like_save", "save", "none"]
            }
        )

        DEBUG = False

        y_preds = {}
        scores = {}
        for k, gbm in gbms.items():
            if k == "baseline":
                gbm.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    callbacks=[ProgressBarCallback(desc=k)],
                )
            else:
                gbm.fit(
                    self.X,
                    self.y,
                    callbacks=[ProgressBarCallback(desc=k)],
                )
            y_pred = gbm.predict(X_test)
            y_preds[k] = y_pred
            scores[k] = mean_squared_error(y_test, y_pred)
            lgb.plot_importance(gbm)
            lgb.plot_split_value_histogram(gbm, feature=0)
            lgb.plot_metric(gbm, metric="l2")
            if DEBUG:
                lgb.plot_tree(gbm, tree_index=0)
            if not DEBUG:
                plt.close("all")
            # There seem to be no way to get the actual trained number of trees
            # without digging into the C++ code.
            # It is too difficult to test if "none" mode works correctly.

        if DEBUG:
            plt.show()

        # "refit" < "refit_like_save" == "save" < "none" < "baseline"
        print(scores)
        self.assertLess(scores["refit"], scores["refit_like_save"])
        self.assertEqual(scores["refit_like_save"], scores["save"])
        self.assertLess(scores["save"], scores["none"])
        self.assertLess(scores["none"], scores["baseline"])
