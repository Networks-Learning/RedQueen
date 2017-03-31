#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='redqueen',
      version='1.0.0',
      description='RedQueen code.',
      author='Utkarsh Upadhyay',
      author_email='mail@musicallyut.in',
      url='https://github.com/Networks-Learning/RedQueen',
      packages=find_packages(),
      install_required=['pandas', 'broadcast', 'numpy', 'decorated_options', 'matplotlib']
)
