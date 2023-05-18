# Usage

## SciKit-Learn API, simple

```python
from lightgbm import LGBMRegressor
from lightgbm_callbacks import LGBMDartEarlyStoppingEstimator
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
LGBMDartEarlyStoppingEstimator(
    LGBMRegressor(boosting_type="dart"), # or "gbdt", ...
    stopping_rounds=10, # or n_iter_no_change=10
    test_size=0.2, # or validation_fraction=0.2
    shuffle=False,
    tqdm_cls="rich", # "auto", "autonotebook", ...
).fit(X_train, y_train)
```

## Scikit-Learn API, manually passing callbacks

```python
from lightgbm import LGBMRegressor
from lightgbm_callbacks import ProgressBarCallback, DartEarlyStoppingCallback
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
early_stopping_callback = DartEarlyStoppingCallback(stopping_rounds=10)
LGBMRegressor(
).fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[
        early_stopping_callback,
        ProgressBarCallback(early_stopping_callback=early_stopping_callback),
    ],
)
```

## Details on `DartEarlyStoppingCallback`

Below is a description of the `DartEarlyStoppingCallback` `method` parameter and `lgb.plot_metric` for each `lgb.LGBMRegressor(boosting_type="dart", n_estimators=1000)` trained with entire `sklearn_datasets.load_diabetes()` dataset.

| Method     | Description                                                                                  | iteration                                                   | Image                                 | Actual iteration |
| ---------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------- | ---------------- |
| (Baseline) | If Early stopping is not used.                                                               | `n_estimators`                                              | ![image](docs/_static/m_baseline.png) | 1000             |
| `"none"`   | Do nothing and return the original estimator.                                                | `min(best_iteration + early_stopping_rounds, n_estimators)` | ![image](docs/_static/m_none.png)     | 50               |
| `"save"`   | Save the best model by deepcopying the estimator and return the best model (using `pickle`). | `min(best_iteration + 1, n_estimators)`                     | ![image](docs/_static/m_save.png)     | 21               |
| `"refit"`  | Refit the estimator with the best iteration and return the refitted estimator.               | `min(best_iteration, n_estimators)`                         | ![image](docs/_static/m_refit.png)    | 20               |
