[![Build Status](https://travis-ci.org/rlirey/psmatching.svg?branch=master)](https://travis-ci.org/rlirey/psmatching)

[![Documentation Status](https://readthedocs.org/projects/psmatching/badge/?version=latest)](https://psmatching.readthedocs.io/en/latest/?badge=latest)

# PSMatching

Ryan L. Irey, M.A., M.S.

Insititute for Health Informatics

University of Minnesota

### Features
`psmatching` is a package for implementing propensity score matching in Python 3.

The following functionality is included in the package:
  - Calculation of propensity scores based on a specified model
  - Matching of _k_ controls to each treatment case
  - Evaluation of the matching process using statistical methods

### Technology

`psmatching` uses a number of open source projects to work properly:

* [`pandas`](https://pandas.pydata.org/)
* [`numpy`](https://www.numpy.org/)
* [`scipy`](https://www.scipy.org/)
* [`statsmodels`](https://www.statsmodels.org/stable/index.html)

`psmatching` itself is open source with a [public repository](https://github.com/rlirey/psmatching) on GitHub.

### Installation
Install `psmatching` via `pip`
```sh
$ pip install psmatching
```
### Example
Coming soon!

### Documentation
Documentation for this package can be found [here](https://psmatching.readthedocs.io/en/latest/).
Documentation is currently under construction!

### Acknowledgments
Parts of this package's source code are modified from Kellie Otto's pscore-match package. More info on that package can be found [here](http://www.kellieottoboni.com/pscore_match/).

### License
Apache 2.0

