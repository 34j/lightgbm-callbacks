from __future__ import annotations

from lightgbm.callback import CallbackEnv


class CallbackBase:
    def _init(self, env: CallbackEnv) -> None:
        pass  # pragma: no cover

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
