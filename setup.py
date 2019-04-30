import os


descr = """Propensity score matching"""


DISTNAME = 'psmatching'
DESCRIPTION = 'Propensity score matching'
LONG_DESCRIPTION = descr
AUTHOR = 'Ryan L. Irey'
AUTHOR_EMAIL = 'ireyx001@umn.edu'
URL = 'http://www.github.com/rlirey/psmatching'
LICENSE = 'Apache 2.0 License'
DOWNLOAD_URL = 'http://www.github.com/rlirey/psmatching'
VERSION = '0.1.dev12'
PYTHON_VERSION = (3, 5)


INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'scipy',
    'statsmodels'
]


if __name__ == "__main__":

    from setuptools import setup
    setup(
        name=DISTNAME,
        version=VERSION,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,

        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],

        install_requires=INSTALL_REQUIRES,

        packages=['psmatching'],
    )
