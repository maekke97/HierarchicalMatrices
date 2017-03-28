from distutils.core import setup

setup(
    name='HierMat',
    version='0.01',
    packages=['HierMat'],
    url='http://hierarchical-matrices.readthedocs.io/en/latest/HierMat.html',
    download_url='https://git.math.uzh.ch/markus.neumann/HMatrix',
    license='GPL',
    author='Markus Neumann',
    author_email='markus.neumann@math.uzh.ch',
    requires=['numpy', 'matplotlib'],
    description='''This package is the result of my master thesis at the Institute of Mathematics, University of Zurich.
It provides a framework for the concept of hierarchical matrices and is manly based on the book by W. Hackbusch.'''
)
