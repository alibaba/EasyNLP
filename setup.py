#!/usr/bin/env python
from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()
setup(
    # Metadata
    name='easynlp',
    version='0.0.3',
    python_requires='>=2.7,>=3.6',
    author='PAI NLP',
    author_email='minghui.qmh@alibaba-inc.com'
    'chengyu.wcy@alibaba-inc.com'
    'huangjun.hj@alibaba-inc.com',
    url='http://gitlab.alibaba-inc.com/groups/PAI-TL',
    description='PAI EasyNLP Toolkit',
    long_description=readme,
    entry_points={'console_scripts': ['easynlp=easynlp.cli:main']},
    long_description_content_type='text/markdown',
    packages=find_packages(),
    license='Apache-2.0',

    #Package info
    install_requires=requirements)
