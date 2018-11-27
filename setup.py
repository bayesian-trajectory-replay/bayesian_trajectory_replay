# -*- coding: utf-8 -*-
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

from setuptools import setup, find_packages
from distutils.extension import Extension

setup(name = 'bayesian_trajectory_replay',
	version = '0.1',
	description = 'Non-parametric Bayesian learning of robot behaviors from demonstration',
	classifiers = [
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: BSD License',
		'Operating System :: OS Independent',
		'Programming Language :: Cython',
		'Programming Language :: Python :: 2 :: Only',
		'Topic :: Scientific/Engineering :: Artificial Intelligence'
	],
	long_description = open('README.md').read(),
	url = 'https://github.com/bayesian_trajectory_replay/bayesian_trajectory_replay',
	author = 'Stéphane Magnenat and Francis Colas',
	author_email = 'francis.colas@inria.fr',
	license = 'BSD',
	packages = find_packages(),
	zip_safe = False,
	install_requires = ['numpy', 'scipy', 'pyyaml'],
	setup_requires = ['setuptools >= 18.0', 'Cython'],
	ext_modules = [ 
		Extension("bayesian_trajectory_replay.btr_helper_gaussian", ["bayesian_trajectory_replay/btr_helper_gaussian.pyx"], extra_compile_args=["-O3"]),
		Extension("bayesian_trajectory_replay.btr_helper_cauchy", ["bayesian_trajectory_replay/btr_helper_cauchy.pyx"], extra_compile_args=["-O3"]),
		Extension("bayesian_trajectory_replay.btr_helper_temporal", ["bayesian_trajectory_replay/btr_helper_temporal.pyx"], extra_compile_args=["-O3"])
	]
)
