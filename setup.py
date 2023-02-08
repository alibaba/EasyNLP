#!/usr/bin/env python
from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()
setup(
    # Metadata
    name='pai-easynlp',
    version='0.1.2',
    python_requires='>=3.6',
    author='PAI NLP',
    author_email='easynlp@list.alibaba-inc.com',
    url='https://github.com/alibaba/EasyNLP',
    description='PAI EasyNLP Toolkit',
    long_description=readme,
    entry_points={'console_scripts': ['easynlp=easynlp.cli:main']},
    long_description_content_type='text/markdown',
    packages=find_packages(),
    license='Apache-2.0',

    #Package info
    install_requires=requirements)
