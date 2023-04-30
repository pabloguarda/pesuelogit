#Compatibility with imports from Python 2.x
from __future__ import absolute_import

"""Top-level package for pesuelogit."""

__author__ = """Pablo Guarda"""
__email__ = 'pabloguarda@cmu.edu'
__version__ = '0.1.0'

import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# print(project_root)

# Modules available for user
import pesuelogit.networks
import pesuelogit.models
import pesuelogit.descriptive_statistics
import pesuelogit.experiments
import pesuelogit.etl
import pesuelogit.visualizations