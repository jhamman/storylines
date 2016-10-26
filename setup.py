#!/usr/bin/env python
import os
import re
import sys
import warnings

from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''


DISTNAME = 'storylines'
LICENSE = 'GNU General Public License v3.0'
AUTHOR = 'Joseph Hamman'
AUTHOR_EMAIL = 'jhamman@ucar.edu'
URL = 'https://github.com/jhamman/storylines'
CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Operating System :: POSIX',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Atmospheric Science',
]

INSTALL_REQUIRES = ['xarray >= 0.8.2']
TESTS_REQUIRE = ['pytest >= 3.0.3']

DESCRIPTION = "Quantitative hydrologic storylines to assess climate impacts"
LONG_DESCRIPTION = """
**storylines** is a framework for characterizing uncertainty in traditional
hydroloic climate impacts modeling chains (climate models, downscaling methods,
hydrologic models). It includes tools for evaluating model fidelity and culling
models accordingly to reduce these uncertainties, and finally distilling
projections into a discrete set of quantitative hydrologic storylines that
represent key, impact-focused, features from the full range of future
scenarios.

**storylines** is being developed at the National Center for Atmospheric
Research (NCAR_), Research Applications Laboratory (RAL_) - Hydrometeorological
Applications Program (HAP_) under the support of USACE.

.. _NCAR: http://ncar.ucar.edu/
.. _RAL: https://www.ral.ucar.edu
.. _HAP: https://www.ral.ucar.edu/hap

Important links
---------------

- HTML documentation: http://storylines.readthedocs.io
- Issue tracker: http://github.com/jhamman/storylines/issues
- Source code: http://github.com/jhamman/storylines
"""

# code to extract and write the version copied from pandas
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('storylines/version.py'):
            warnings.warn("Couldn't get git revision, using existing"
                          "storylines/version.py")
            write_version = False
        else:
            warnings.warn(
                "Couldn't get git revision, using generic version string")
    else:
        # have git, in git dir, but may have used a shallow clone
        # (travis does this)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')

else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'storylines', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

if write_version:
    write_version_py()

setup(name=DISTNAME,
      version=FULLVERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages(),
      package_data={'storylines': ['test/data/*']})
