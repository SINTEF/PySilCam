# -*- coding: utf-8 -*-
import re
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
from pysilcam import __version__


REQUIRES = [
    'docopt',
    'pymba',
]

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
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
    dependency_links=['https://github.com/mabl/pymba@python3'],
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
            'silcam-aquire = pysilcam.__main__:silcam_aquire',
            'silcam-process-rt = pysilcam.__main__:silcam_process_realtime',
            'silcam-process-batch = pysilcam.__main__:silcam_process_batch',
        ]
    },
    tests_require=['pytest'],
    cmdclass={'test': PyTest}
)
