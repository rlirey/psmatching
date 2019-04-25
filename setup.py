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
VERSION = '0.1dev'
PYTHON_VERSION = (3, 5)


INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'dask[complete]',
    'scipy',
    'statsmodels'
]


def write_version_py(filename='psmatching/version.py'):
    template = """# THIS FILE IS GENERATED FROM THE psmatching setup.py version='%s'"""

    try:
        fname = os.path.join(os.path.dirname(__file__), filename)
        with open(fname, 'w') as f:
            f.write(template % VERSION)
    except IOError:
        raise IOError("Could not open/write to psmatching/version.py - did you "
                      "install using sudo in the past? If so, run\n"
                      "sudo chown -R your_username ./*\n"
                      "from package root to fix permissions, and try again.")


if __name__ == "__main__":

    write_version_py()

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
            'License :: OSI Approved :: Apache 2.0 License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],

        install_requires=INSTALL_REQUIRES,

        packages=['psmatching'],
    )
