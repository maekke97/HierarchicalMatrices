"""
HierMat
=======

This library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this package.  If not, see <http://www.gnu.org/licenses/>.
"""
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HierMat.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

with open('README.rst', 'r') as readme:
    long_desc = readme.read()

setup(
    name='HierMat',
    version='0.6.8',
    packages=['HierMat'],
    url='http://hierarchicalmatrices.readthedocs.io/en/latest/index.html',
    download_url='https://github.com/maekke97/HierarchicalMatrices',
    license='GNU GPL v3',
    author='Markus Neumann',
    author_email='markus.neumann@math.uzh.ch',
    requires=['numpy', 'matplotlib', 'scipy'],
    description='Framework for Hierarchical Matrices',
    long_description=long_desc
)
