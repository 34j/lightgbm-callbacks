from __future__ import annotations

import pickle  # nosec
from functools import partial
from typing import Any, Callable, List, Tuple, TypeVar, Union

from lightgbm.basic import _ConfigAliases, _log_info, _log_warning
from lightgbm.callback import CallbackEnv, EarlyStopException, _format_eval_result

_LGBM_BoosterEvalMethodResultType = Tuple[str, str, float, bool]
_ListOfEvalResultTuples = Union[
    List[_LGBM_BoosterEvalMethodResultType], List[Tuple[str, str, float, bool, float]]
]

TObj = TypeVar("TObj", bound=Any)


def _complete_deepcopy(obj: TObj) -> TObj:
    return pickle.loads(pickle.dumps(obj))  # nosec


def _gt_delta(curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score > best_score + delta


def _lt_delta(curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score < best_score - delta


def _is_train_set(ds_name: str, eval_name: str, train_name: str) -> bool:
    return (ds_name == "cv_agg" and eval_name == "train") or ds_name == train_name


class EarlyStoppingCallback:
    """A callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score doesn't improve by at least ``min_delta``.
    Validation score needs to improve at least every ``stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    The index of iteration that has the best performance will be saved in the ``best_iteration``
    attribute of a model.

    Compared to the official implementation, the best_iteration information
    is retained even when using dart.

    Parameters
    ----------
    stopping_rounds : int
        The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
        Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to log message with early stopping information.
        By default, standard output resource is used.
        Use ``register_logger()`` function to register a custom logger.
    min_delta : float or list of float, optional (default=0.0)
        Minimum improvement in score to keep training.
        If float, this single value is used for all metrics.
        If list, its length should match the total number of metrics."""

    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool = False,
        verbose: bool = True,
        min_delta: float | list[float] = 0.0,
    ) -> None:
        self.order = 30
        self.before_iteration = False

        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta

        self.enabled = True
        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score: list[float] = []
        self.best_iter: list[int] = []
        self.best_score_list: list[_ListOfEvalResultTuples] = []
        self.cmp_op: list[Callable[[float, float], bool]] = []
        self.first_metric = ""

    def _init(self, env: CallbackEnv) -> None:
        is_dart = any(
            env.params.get(alias, "") == "dart"
            for alias in _ConfigAliases.get("boosting")
        )
        self.is_dart = is_dart
        only_train_set = len(env.evaluation_result_list) == 1 and _is_train_set(
            ds_name=env.evaluation_result_list[0][0],
            eval_name=env.evaluation_result_list[0][1].split(" ")[0],
            train_name=env.model._train_data_name,
        )
        self.enabled = (
            not only_train_set
        )  # not is_dart and not only_train_set # NOTE: added by 34j
        if not self.enabled:
            if is_dart:
                _log_warning("Early stopping is not available in dart mode")
            elif only_train_set:
                _log_warning("Only training set found, disabling early stopping.")
            return
        if not env.evaluation_result_list:
            raise ValueError(
                "For early stopping, "
                "at least one dataset and eval metric is required for evaluation"
            )

        if self.stopping_rounds <= 0:
            raise ValueError("stopping_rounds should be greater than zero.")

        if self.verbose:
            _log_info(
                f"Training until validation scores don't improve for {self.stopping_rounds} rounds"
            )

        self._reset_storages()

        n_metrics = len({m[1] for m in env.evaluation_result_list})
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError(
                    "Values for early stopping min_delta must be non-negative."
                )
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info("Disabling min_delta for early stopping.")
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(
                        f"Using {self.min_delta[0]} as min_delta for all metrics."
                    )
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError(
                        "Must provide a single value for min_delta or as many as metrics."
                    )
                if self.first_metric_only and self.verbose:
                    _log_info(
                        f"Using only {self.min_delta[0]} as early stopping min_delta."
                    )
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError("Early stopping min_delta must be non-negative.")
            if (
                self.min_delta > 0
                and n_metrics > 1
                and not self.first_metric_only
                and self.verbose
            ):
                _log_info(f"Using {self.min_delta} as min_delta for all metrics.")
            deltas = [self.min_delta] * n_datasets * n_metrics

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        self.first_metric = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            self.best_iter.append(0)
            if eval_ret[3]:  # greater is better
                self.best_score.append(float("-inf"))
                self.cmp_op.append(partial(_gt_delta, delta=delta))
            else:
                self.best_score.append(float("inf"))
                self.cmp_op.append(partial(_lt_delta, delta=delta))

    def _final_iteration_check(
        self, env: CallbackEnv, eval_name_splitted: list[str], i: int
    ) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                best_score_str = "\t".join(
                    [
                        _format_eval_result(x, show_stdv=True)
                        for x in self.best_score_list[i]
                    ]
                )
                _log_info(
                    "Did not meet early stopping. "
                    f"Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}"
                )
                if self.first_metric_only:
                    _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
            if self.is_dart:  # NOTE: added by 34j
                # return current iter
                raise EarlyStopException(env.iteration, self.best_score_list[i])
            raise EarlyStopException(self.best_iter[i], self.best_score_list[i])

    # NOTE: added by 34j
    def on_best_score_updated(self, env: CallbackEnv, i: int) -> None:
        pass

    def on_best_score_not_updated(self, env: CallbackEnv, i: int) -> None:
        pass

    def on_best_score_update_started(self, env: CallbackEnv, i: int) -> None:
        pass

    def on_best_score_update_stopped(self, env: CallbackEnv, i: int) -> None:
        pass

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return

        # self.best_score_list is initialized to an empty list
        first_time_updating_best_score_list = self.best_score_list == []
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if first_time_updating_best_score_list or self.cmp_op[i](
                score, self.best_score[i]
            ):
                # NOTE: added by 34j
                if self.best_iter[i] != env.iteration - 1:
                    self.on_best_score_update_started(env, i)
                self.on_best_score_updated(env, i)

                self.best_score[i] = score
                self.best_iter[i] = env.iteration
                if first_time_updating_best_score_list:
                    self.best_score_list.append(env.evaluation_result_list)
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            else:
                # NOTE: added by 34j
                if self.best_iter[i] == env.iteration - 1:
                    self.on_best_score_update_stopped(env, i)
                self.on_best_score_not_updated(env, i)
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if self.first_metric_only and self.first_metric != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if _is_train_set(
                env.evaluation_result_list[i][0],
                eval_name_splitted[0],
                env.model._train_data_name,
            ):
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = "\t".join(
                        [
                            _format_eval_result(x, show_stdv=True)
                            for x in self.best_score_list[i]
                        ]
                    )
                    _log_info(
                        f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t"
                        f"{eval_result_str}"
                    )
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
                if self.is_dart:  # NOTE: added by 34j
                    # return current iter
                    raise EarlyStopException(env.iteration, self.best_score_list[i])
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            self._final_iteration_check(env, eval_name_splitted, i)


class DartEarlyStoppingCallback(EarlyStoppingCallback):
    """A callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score doesn't improve by at least ``min_delta``.
    Validation score needs to improve at least every ``stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    The index of iteration that has the best performance will be saved in the ``best_iteration``
    attribute of a model.

    When using dart, the model is copied and retained using pickle at iterations
    where scores no longer improve. Eventually, the number of iterations of the ``best_model``
    should be ``max(best_iteration[i] + 2, num_boost_round/n_iter/n_estimators)``.
    (``max(best_iteration[i] + 1, num_boost_round/n_iter/n_estimators)`` in case of normal early stopping.)

    Parameters
    ----------
    stopping_rounds : int
        The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
        Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to log message with early stopping information.
        By default, standard output resource is used.
        Use ``register_logger()`` function to register a custom logger.
    min_delta : float or list of float, optional (default=0.0)
        Minimum improvement in score to keep training.
        If float, this single value is used for all metrics.
        If list, its length should match the total number of metrics."""

    def _reset_storages(self) -> None:
        self.best_model: list[Any] = []
        return super()._reset_storages()

    def _save_model(self, env: CallbackEnv, i: int) -> None:
        # _log_info(f"Saving best model for metric {i}.")
        self.best_model.extend([None] * (i + 1 - len(self.best_model)))
        self.best_model[i] = _complete_deepcopy(env.model)

    def on_best_score_update_stopped(self, env: CallbackEnv, i: int) -> None:
        if not self.is_dart:
            return

        self._save_model(env, i)

    def _final_iteration_check(
        self, env: CallbackEnv, eval_name_splitted: list[str], i: int
    ) -> None:
        if not self.is_dart:
            return super()._final_iteration_check(env, eval_name_splitted, i)

        if env.iteration == env.end_iteration - 1:
            self._save_model(env, i)
        return super()._final_iteration_check(env, eval_name_splitted, i)
