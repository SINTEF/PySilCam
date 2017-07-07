# -*- coding: utf-8 -*-
import re
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
from pysilcam import __version__


REQUIRES = [
    'docopt',
    'configparser',
    'numpy',
    'pandas',
    'matplotlib',
    'imageio',
    'scikit-image',
#    'pymba',
]

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        params = {"args":self.test_args}
        params["args"] +=  ["--junitxml", "test-report/output.xml"]
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


setup(
    name='PySilCam',
    version='0.1.0',
    description='A Python interface to the SilCam.',
    long_description=read('README.rst'),
    author='Emlyn Davies',
    author_email='emlyn.davies@sintef.no',
    install_requires=REQUIRES,
    # Use Python 3 branch on alternate repo for Pymba
    #dependency_links=['git+https://github.com/mabl/pymba@python3'],
    # Use master branch for Pymba on Python 2
    #dependency_links=['git+https://github.com/morefigs/pymba@python3'],
    license=read('LICENSE'),
    zip_safe=False,
    keywords='silcam',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['pysilcam'],
    entry_points={
        'console_scripts': [
            'silcam-acquire = pysilcam.__main__:silcam_acquire',
            'silcam-live = pysilcam.__main__:silcam_acquire',
            'silcam-process-batch = pysilcam.__main__:silcam_process_batch',
        ]
    },
    tests_require=['pytest'],
    cmdclass={'test': PyTest}
)
