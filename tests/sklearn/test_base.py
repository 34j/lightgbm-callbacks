from unittest import TestCase

from lightgbm import LGBMRegressor

from lightgbm_callbacks.sklearn import EstimatorWrapperBase


class TestEstimatorWrapperBase(TestCase):
    def test_isinstance(self) -> None:
        self.assertIsInstance(
            EstimatorWrapperBase(LGBMRegressor()), EstimatorWrapperBase
        )
        self.assertIsInstance(EstimatorWrapperBase(LGBMRegressor()), LGBMRegressor)

    def test_issubclass(self) -> None:
        self.assertFalse(issubclass(EstimatorWrapperBase, LGBMRegressor))
        self.assertTrue(issubclass(EstimatorWrapperBase, EstimatorWrapperBase))
