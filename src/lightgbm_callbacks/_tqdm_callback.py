from __future__ import annotations

import importlib
import warnings
from collections import OrderedDict
from typing import Any, Literal

import tqdm
from lightgbm.callback import CallbackEnv

from ._base import CallbackBase


class ProgressBarCallback(CallbackBase):
    tqdm_cls: type[tqdm.std.tqdm] | None
    pbar: tqdm.std.tqdm | None

    def __init__(
        self,
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
        | None = "auto",
        early_stopping_callback: Any | None = None,
        **tqdm_kwargs: Any,
    ) -> None:
        """Progress bar callback for LightGBM training.

        Parameters
        ----------
        tqdm_cls : Literal['auto', 'autonotebook', 'std', 'notebook', 'asyncio',
        'keras', 'dask', 'tk', 'gui', 'rich', 'contrib.slack', 'contrib.discord',
        'contrib.telegram', 'contrib.bells'] or type[tqdm.std.tqdm] or None, optional
            The tqdm class or module name, by default 'auto'
        early_stopping_callback : Any | None, optional
            The early stopping callback, by default None
        **tqdm_kwargs : Any
            The keyword arguments passed to the tqdm class initializer

            .. rubric:: Example

            .. code-block:: python
                early_stopping_callback = early_stopping(stopping_rounds=50)
                callbacks = [
                    early_stopping_callback,
                    ProgressBarCallback(early_stopping_callback=early_stopping_callback)
                ]
                estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        """
        self.order = 40
        self.before_iteration = False
        if isinstance(tqdm_cls, str):
            tqdm_module = importlib.import_module(f"tqdm.{tqdm_cls}")
            self.tqdm_cls = getattr(tqdm_module, "tqdm")
        else:
            self.tqdm_cls = tqdm_cls
        self.early_stopping_callback = early_stopping_callback
        self.tqdm_kwargs = tqdm_kwargs
        if "total" in tqdm_kwargs:
            warnings.warn("'total' in tqdm_kwargs is ignored.", UserWarning)
        self.pbar = None

    def _init(self, env: CallbackEnv) -> None:
        # create pbar on first call
        if self.tqdm_cls is None:
            return
        tqdm_kwargs = self.tqdm_kwargs.copy()
        tqdm_kwargs["total"] = env.end_iteration - env.begin_iteration
        self.pbar = self.tqdm_cls(**tqdm_kwargs)

    def __call__(self, env: CallbackEnv) -> None:
        super().__call__(env)
        if self.pbar is None:
            return

        # update postfix
        if len(env.evaluation_result_list) > 0:
            # If OrderedDict is not used, the order of display is disjointed and slightly difficult to see.
            # https://github.com/microsoft/LightGBM/blob/a97c444b4cf9d2755bd888911ce65ace1fe13e4b/python-package/lightgbm/callback.py#L56-66
            if self.early_stopping_callback is not None:
                postfix = OrderedDict(
                    [
                        (
                            f"{entry[0]}'s {entry[1]}",
                            f"{entry[2]:g}{'=' if entry[2] == best_score else ('>' if cmp_op else '<')}"
                            f"{best_score:g}@{best_iter}it",
                        )
                        for entry, cmp_op, best_score, best_iter in zip(
                            env.evaluation_result_list,
                            self.early_stopping_callback.cmp_op,
                            self.early_stopping_callback.best_score,
                            self.early_stopping_callback.best_iter,
                        )
                    ]
                )
            else:
                postfix = OrderedDict(
                    [
                        (f"{entry[0]}'s {entry[1]}", f"{entry[2]:g}")
                        for entry in env.evaluation_result_list
                    ]
                )
            self.pbar.set_postfix(ordered_dict=postfix, refresh=False)

        self.pbar.n += 1
        self.pbar.refresh()
