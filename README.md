This repository contains the Python/Cython core of our work on non-parametric Bayesian learning of robot behaviors from demonstration.

Stéphane Magnenat and Francis Colas.
Copyright (c) 2011-2018 ETH Zurich.
This work was mostly done at the Autonomous Systems Lab (http://www.asl.ethz.ch/).

# Install

## PyPI

You can install using `pip` with this command:

     python2 -m pip install bayesian-trajectory-replay

## By hand

You can compile and install the package with `python2 -m pip install .` in the this directory.

# Test

You can run `test.py` from within the `test` directory to see the algorithm in action on a simplistic case.

# Cite

This is the code corresponding to [this paper](https://link.springer.com/article/10.1007/s10514-021-10019-4) ([pre-print PDF](https://stephane.magnenat.net/publications/A%20Bayesian%20Tracker%20for%20Synthesizing%20Mobile%20Robot%20Behaviour%20from%20Demonstration%20-%20Magnenat%20and%20Colas%20-%20Autonomous%20Robots%20-%202021.pdf)).

You can cite it:
```
@article{magnenat2021bayesian,
  title={A Bayesian tracker for synthesizing mobile robot behaviour from demonstration},
  author={Magnenat, St{\'e}phane and Colas, Francis},
  journal={Autonomous Robots},
  volume={45},
  number={8},
  pages={1077--1096},
  year={2021},
  publisher={Springer}
}
```
