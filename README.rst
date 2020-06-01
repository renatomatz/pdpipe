pdpipe ˨ 
########

|PyPI-Status| |Downloads| |PyPI-Versions| |Build-Status| |Codecov| |Codefactor| |LICENCE|


Easy pipelines for pandas DataFrames (`learn how! <https://tirthajyoti.github.io/Notebooks/Pandas-pipeline-with-pdpipe>`_).

Website: `https://pdpipe.github.io/pdpipe/ <https://pdpipe.github.io/pdpipe/>`_

Documentation: `https://pdpipe.github.io/pdpipe/doc/pdpipe/ <https://pdpipe.github.io/pdpipe/doc/pdpipe/>`_

.. code-block:: python

  >>> df = pd.DataFrame(
          data=[[4, 165, 'USA'], [2, 180, 'UK'], [2, 170, 'Greece']],
          index=['Dana', 'Jane', 'Nick'],
          columns=['Medals', 'Height', 'Born']
      )
  >>> import pdpipe as pdp
  >>> pipeline = pdp.ColDrop('Medals').OneHotEncode('Born')
  >>> pipeline(df)
              Height  Born_UK  Born_USA
      Dana     165        0         1
      Jane     180        1         0
      Nick     170        0         0

.. .. alternative symbols: ˨ ᛪ ᛢ ᚶ ᚺ ↬ ⑀ ⤃ ⤳ ⥤ 』

.. contents::

.. section-numbering::

Documentation
=============

This is the repository of the ``pdpipe`` package, and this readme file is aimed to help potential contributors to the project.

To learn more about how to use ``pdpipe``, either `visit pdpipe's homepage <https://pdpipe.github.io/pdpipe/>`_ or read the `online documentation of pdpipe <https://pdpipe.github.io/pdpipe/doc/pdpipe/>`_.


Installation
============

Install ``pdpipe`` with:

.. code-block:: bash

  pip install pdpipe

Some pipeline stages require ``scikit-learn``; they will simply not be loaded if ``scikit-learn`` is not found on the system, and ``pdpipe`` will issue a warning. To use them you must also `install scikit-learn <http://scikit-learn.org/stable/install.html>`_.


Similarly, some pipeline stages require ``nltk``; they will simply not be loaded if ``nltk`` is not found on your system, and ``pdpipe`` will issue a warning. To use them you must additionally `install nltk <http://www.nltk.org/install.html>`_.



Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed, especially since this package is very much in its infancy and many other pipeline stages can be added.

Installing for development
--------------------------

Clone:

.. code-block:: bash

  git clone git@github.com:shaypal5/pdpipe.git


Install in development mode with test dependencies:

.. code-block:: bash

  cd pdpipe
  pip install -e ".[test]"


Running the tests
-----------------

To run the tests, use:

.. code-block:: bash

  python -m pytest


Notice ``pytest`` runs are configured by the ``pytest.ini`` file. Read it to understand the exact ``pytest`` arguments used.


Adding tests
------------

At the time of writing, ``pdpipe`` is maintained with a test coverage of 100%. Although challenging, I hope to maintain this status. If you add code to the package, please make sure you thoroughly test it. Codecov automatically reports changes in coverage on each PR, and so PR reducing test coverage will not be examined before that is fixed.

Tests reside under the ``tests`` directory in the root of the repository. Each model has a separate test folder, with each class - usually a pipeline stage - having a dedicated file (always starting with the string "test_") containing several tests (each a global function starting with the string "test_"). Please adhere to this structure, and try to separate tests cases to different test functions; this allows us to quickly focus on problem areas and use cases. Thank you! :)


Adding documentation
--------------------

This project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings (in my personal opinion, of course). When documenting code you add to this project, please follow `these conventions`_.

.. _`numpy docstring conventions`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _`these conventions`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Additionally, if you update this ``README.rst`` file,  use ``python setup.py checkdocs`` to validate it compiles.


Adding doctests
---------------

Please notice that for ``pdoc3`` - the Python package used to generate the html documentation files for ``pdpipe`` - to successfully include doctests in the generated documentation files, the whole doctest must be indented in relation to the opening multi-string indentation, like so:

.. code-block:: python


    class ApplyByCols(PdPipelineStage):
        """A pipeline stage applying an element-wise function to columns.

        Parameters
        ----------
        columns : str or list-like
            Names of columns on which to apply the given function.
        func : function
            The function to be applied to each element of the given columns.
        result_columns : str or list-like, default None
            The names of the new columns resulting from the mapping operation. Must
            be of the same length as columns. If None, behavior depends on the
            drop parameter: If drop is True, the name of the source column is used;
            otherwise, the name of the source column is used with the suffix
            '_app'.
        drop : bool, default True
            If set to True, source columns are dropped after being mapped.
        func_desc : str, default None
            A function description of the given function; e.g. 'normalizing revenue
            by company size'. A default description is used if None is given.


        Example
        -------
            >>> import pandas as pd; import pdpipe as pdp; import math;
            >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
            >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
            >>> round_ph = pdp.ApplyByCols("ph", math.ceil)
            >>> round_ph(df)
               ph  lbl
            1   4  acd
            2   8  alk
            3  13  alk
        """


Credits
=======
Created by Shay Palachy  (shay.palachy@gmail.com).

.. alternative:
.. https://badge.fury.io/py/yellowbrick.svg

.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pdpipe.svg
  :target: https://pypi.org/project/pdpipe

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pdpipe.svg
   :target: https://pypi.org/project/pdpipe

.. |Build-Status| image:: https://travis-ci.org/pdpipe/pdpipe.svg?branch=master
  :target: https://travis-ci.org/pdpipe/pdpipe

.. |LICENCE| image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://pypi.python.org/pypi/pdpipe
  
.. .. |LICENCE| image:: https://github.com/shaypal5/pdpipe/blob/master/mit_license_badge.svg
  :target: https://pypi.python.org/pypi/pdpipe
  
.. https://img.shields.io/pypi/l/pdpipe.svg

.. |Codecov| image:: https://codecov.io/github/pdpipe/pdpipe/coverage.svg?branch=master
   :target: https://codecov.io/github/pdpipe/pdpipe?branch=master

  
.. |Codacy|  image:: https://api.codacy.com/project/badge/Grade/7d605e063f114ecdb5569266bd0226cd
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/shaypal5/pdpipe?utm_source=github.com&utm_medium=referral&utm_content=shaypal5/pdpipe&utm_campaign=Badge_Grade_Dashboard

.. |Requirements| image:: https://requires.io/github/shaypal5/pdpipe/requirements.svg?branch=master
     :target: https://requires.io/github/shaypal5/pdpipe/requirements/?branch=master
     :alt: Requirements Status

.. |Downloads| image:: https://pepy.tech/badge/pdpipe
     :target: https://pepy.tech/project/pdpipe
     :alt: PePy stats
     
.. |Codefactor| image:: https://www.codefactor.io/repository/github/pdpipe/pdpipe/badge?style=plastic
     :target: https://www.codefactor.io/repository/github/pdpipe/pdpipe
     :alt: Codefactor code quality
