#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import os

def read_requirements(filename):
    with open(os.path.join(filename)) as fp:
        return fp.read().strip().splitlines()

# with open('README.rst') as readme_file:
#     readme = readme_file.read()
#
# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

setup_requirements = ['pytest-runner', ]

setup(
    author="Pablo Guarda",
    author_email='pabloguarda@cmu.edu',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    description="Parameter estimation of logit-based stochastic user equilibrium (pesuelogit)",
    entry_points={
        'console_scripts': [
            'pesuelogit=pesuelogit.cli:main',
        ],
    },
    license="MIT license",
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='network modelling, computational graphs',
    name='pesuelogit',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    setup_requires=setup_requirements,
    test_suite='tests',
    install_requires=read_requirements("requirements.txt"),
    # test_requires=read_requirements("dev.txt"),
    url='https://github.com/pabloguarda/pesuelogit',
    version='0.1.0',
    zip_safe=False,
)
