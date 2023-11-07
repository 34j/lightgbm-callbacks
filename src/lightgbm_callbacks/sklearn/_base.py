from typing import Any, Generic, TypeVar

from sklearn.base import BaseEstimator, RegressorMixin

TEstimator = TypeVar("TEstimator", bound=Any)


class EstimatorWrapperBase(BaseEstimator, RegressorMixin, Generic[TEstimator]):
    """A base class for estimator wrappers that delegates all attributes to the wrapped estimator."""

    def __init__(self, estimator: TEstimator) -> None:
        """A base class for estimator wrappers that delegates all attributes to the wrapped estimator.

        Parameters
        ----------
        estimator : Any
            The estimator to be wrapped.
        """
        self.estimator = estimator

    def __getattribute__(self, __name: str) -> Any:
        try:
            return object.__getattribute__(self, __name)
        except AttributeError:
            return getattr(self.estimator, __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        try:
            return object.__setattr__(self, __name, __value)
        except AttributeError:
            return setattr(self.estimator, __name, __value)

    def __delattr__(self, __name: str) -> None:
        try:
            return object.__delattr__(self, __name)
        except AttributeError:
            return delattr(self.estimator, __name)

    def __getitem__(self, __key: str) -> Any:
        return self.estimator.__getitem__(__key)

    # Due to the bug in Python, __instancecheck__ does not work.
    # https://bugs.python.org/issue35083

    # This makes the class sklearn.clone() incompatible.
    """@property
    def __class__(self) -> Any:
        return self.estimator.__class__

    @__class__.setter
    def __class__(self, __class__: Any) -> None:  # noqa
        self.estimator.__class__ = __class__"""
