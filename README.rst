Parameter Estimation of Stochastic User Equilibrium with LOGIT assignment models (pesuelogit)
==============================================================================

This codebase allows the joint estimation of the Origin-Destination matrix using system-level data collected in multiple hourly periods and days.  To understand the theory behind the algorithms and the use cases of this tool, you can review the following resources:

+ Preprint: https://dx.doi.org/10.2139/ssrn.4490930
+ Journal article: http://dx.doi.org/10.1016/j.trc.2023.104409

The folder ``examples/notebooks`` contains Jupyter notebooks with code demonstrations that can be reproduced from your local environment. Please follow the instructions below. 

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

Collaboration
=============

For any questions or interest in collaborating on this project, please contact pabloguarda@cmu.edu. This package was developed under the guidance of Prof. Sean Qian. 

Notes
=====
.. _openai-website:
This project is being extended in `nesuelogit <https://github.com/pabloguarda/nesuelogit>`_
