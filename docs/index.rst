PSMatching
==========


.. image:: https://travis-ci.org/rlirey/psmatching.svg?branch=master
   :target: https://travis-ci.org/rlirey/psmatching
   :alt: Build Status


.. image:: https://coveralls.io/repos/github/rlirey/psmatching/badge.svg?branch=master
   :target: https://coveralls.io/github/rlirey/psmatching?branch=master
   :alt: Coverage Status
   
.. image:: https://readthedocs.org/projects/psmatching/badge/?version=latest
   :target: https://psmatching.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


Features
^^^^^^^^

``psmatching`` is a package for implementing propensity score matching in Python 3.

The following functionality is included in the package:


* Calculation of propensity scores based on a specified model
* Matching of *k* controls to each treatment case
* Use of a caliper to control the maximum difference between propensity scores
* Matching with or without replacement
* Evaluation of the matching process using statistical methods


Installation
^^^^^^^^^^^^

Install ``psmatching`` via ``pip``

.. code-block:: sh

   $ pip install psmatching

Usage
^^^^^^^

.. code-block:: py

   >>> # Instantiate PSMatch object
   >>> m = PSMatch(path, model, k)
   
   >>> # Calculate propensity scores and prepare data for matching
   >>> m.prepare_data()
   
   >>> # Perform matching
   >>> m.match(caliper = None, replace = False)
   
   >>> # Evaluate matches via chi-square test
   >>> m.evaluate()


License
^^^^^^^

Apache 2.0
