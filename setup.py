# -*- coding: utf-8 -*-
from distutils.core import setup


version = '1.0.0'

setup(
    name = 'mlencoders',
    packages = ['mlencoders'],
    version = version,
    description = 'Machine Learning encoders for feature transform and engineering.',
    url = 'https://github.com/tcassou/mlencoders',
    download_url = 'https://github.com/tcassou/mlencoders/archive/{}.tar.gz'.format(version),
    keywords = ['machine', 'learning', 'encoder', 'python', 'feature', 'transform', 'engineering'],
    classifiers = [
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=[
        'numpy>=1.14.0',
        'pandas>=0.22.0',
    ],
)
