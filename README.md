# LightGBM Callbacks

<p align="center">
  <a href="https://github.com/34j/lightgbm-callbacks/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/lightgbm-callbacks/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://lightgbm-callbacks.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/lightgbm-callbacks.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/lightgbm-callbacks">
    <img src="https://img.shields.io/codecov/c/github/34j/lightgbm-callbacks.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/lightgbm-callbacks/">
    <img src="https://img.shields.io/pypi/v/lightgbm-callbacks.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/lightgbm-callbacks.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/lightgbm-callbacks.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://lightgbm-callbacks.readthedocs.io" target="_blank">https://lightgbm-callbacks.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/lightgbm-callbacks" target="_blank">https://github.com/34j/lightgbm-callbacks </a>

---

A collection of [LightGBM](https://github.com/microsoft/LightGBM) [callbacks](https://lightgbm.readthedocs.io/en/latest/Python-API.html#callbacks).
Provides implementations of `ProgressBarCallback` ([#5867](https://github.com/microsoft/LightGBM/pull/5867)) and `DartEarlyStoppingCallback` ([#4805](https://github.com/microsoft/LightGBM/issues/4805)), as well as an `LGBMDartEarlyStoppingEstimator` that automatically passes these callbacks. ([#3313](https://github.com/microsoft/LightGBM/issues/3313), [#5808](https://github.com/microsoft/LightGBM/pull/5808))

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install lightgbm-callbacks
```
