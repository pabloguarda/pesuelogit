Parameter Estimation of Stochastic User Equilibrium with LOGIT assignment models (pesuelogit)
==============================================================================

Preprint available at https://dx.doi.org/10.2139/ssrn.4490930

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

If using Linux or a Macbook with Apple Silicon chip, we recommend using Conda for python packaging and dependency management. The steps are:

1. Clone this repository.

2. Download and install Anaconda: https://docs.anaconda.com/anaconda/install/index.html
3. Create virtual environment: ``conda create -n pesuelogit``
4. Activate environment: ``conda activate pesuelogit``
5. Install dependencies: ``conda env update -f pesuelogit.yml``. If you are using linux, use ``pesuelogit-linux.yml`` instead.
6. Run the tests: `pytest`

This repository is currently compatible with Python 3.9.x


