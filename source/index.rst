.. Hierarchical Matrices documentation master file, created by
   sphinx-quickstart on Thu Mar 23 11:20:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hierarchical Matrices's documentation!
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Documentation
=============

:Author: `Markus Neumann`_

:Date: |today|

:Supervisor: `Prof. Dr. Stefan Sauter`_


This package is the result of my master thesis at the `Institute of Mathematics, University of Zurich`_.

It provides an interface in python to the concept of hierarchical matrices as described by :cite:`hackbusch2015hierarchical`.

The main goal is to provide an easy to use yet performing alternative to existing implementations in C

* Hosted on GitHub: `GitHub <https://github.com/maekke97/HierarchicalMatrices>`_
* Available on PyPi: `PyPi <https://pypi.python.org/pypi/HierMat>`_
* Documentation: `ReadTheDocs <http://hierarchicalmatrices.readthedocs.io/en/latest>`_
* Continuous Integration: `Travis CI <https://travis-ci.org/maekke97/HierarchicalMatrices>`_
* Code Coverage: `Coveralls <https://coveralls.io/github/maekke97/HierarchicalMatrices>`_
* Code Quality: `SonarQube <https://sonarqube.com/dashboard/index?id=hierarchicalmatrices>`_

.. image:: https://readthedocs.org/projects/hierarchicalmatrices/badge/

.. image:: https://travis-ci.org/maekke97/HierarchicalMatrices.svg?branch=master

.. image:: https://coveralls.io/repos/github/maekke97/HierarchicalMatrices/badge.svg?branch=master

.. image:: https://sonarqube.com/api/badges/gate?key=hierarchicalmatrices

.. image:: https://img.shields.io/pypi/v/HierMat.svg

Installation
------------

The package is available on different platforms, so there are different ways to install:

#. The easiest would be using ``pip``:

   .. code-block:: bash

      pip install [--user -U] HierMat

#. Another way is to download the source package e.g. from `GitHub <https://github.com/maekke97/HierarchicalMatrices>`_
   and install from there:

   .. code-block:: bash

      wget https://github.com/maekke97/HierarchicalMatrices/archive/master.zip
      unzip master.zip
      cd HierarchicalMatrices-master
      python setup.py install

Basic Usage
-----------
Best look at the examples section

.. code-block:: python

   from HierMat import *
   model_1d(32, 2, 1)

Links and references
--------------------

.. rubric:: Links

.. _Institute of Mathematics, University of Zurich: http://www.math.uzh.ch/index.php?&L=1

`Institute of Mathematics, University of Zurich`_

.. _Prof. Dr. Stefan Sauter: http://www.math.uzh.ch/index.php?professur&key1=105&L=1

`Prof. Dr. Stefan Sauter`_

.. _GitHub Repository: https://github.com/maekke97/HierarchicalMatrices

`GitHub Repository`_

.. _Markus Neumann: markus.neumann@math.uzh.ch

.. rubric:: References

.. todo:: Add thesis to bib

.. bibliography:: thesis.bib
