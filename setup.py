from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A package to train auto-encoders to learn CV on certain 2D potentials'
LONG_DESCRIPTION = 'This package allows to use some simple 2D potentials to analyse and test the behavior of various machine learning approaches to identify collective variables'

setup(
    name="2dAEandCVs",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="<Thomas Pigeon>",
    author_email="<thom.pigeon@gmail.com>",
    license='',
    packages=find_packages(),
    install_requires=[],
    keywords='Auto-encoders, Reaction coordinates, Collective variables',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
