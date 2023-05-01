Parameter Estimation of Stochastic User Equilibrium with LOGIT assignment models (pesuelogit)
==============================================================================


Development Setup
=================

If using a Windows computer or a Macbook with Intel chip, you can use poetry_ for python packaging and dependency management. The steps are:

1. Clone this repository.
2. `Install <https://python-poetry.org/docs/#installation>`_  the poetry cli.
3. Create a virtual environment: ``poetry shell``
4. Create .lock file with project dependencies: ``poetry lock``.
5. Install project dependencies: ``poetry install``.
6. Run the tests: `pytest`

.. _poetry: https://python-poetry.org/

If using a Macbook with Apple Silicon chip, we recommend using Conda for python packaging and dependency management. The steps are:

1. Clone this repository.

2. Download and install Anaconda for 64-Bit (M1): https://docs.anaconda.com/anaconda/install/index.html
3. Create virtual environment: ``conda create -n pesuelogit``
4. Activate environment: ``conda activate pesuelogit``
5. Install dependencies: ``conda env update -f pesuelogit.yml``
6. Run the tests: `pytest`

This repository is currently compatible with Python 3.9.x


