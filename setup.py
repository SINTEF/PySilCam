# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
import distutils.cmd

REQUIRES = [
    'docopt==0.6.2',
    'configparser==3.5.0',
    'numpy==1.15.2',
    'pandas==0.20.3',
    'xlrd',
    'openpyxl==2.4.8',
    'matplotlib==3.0.0',
    'imageio==2.4.1',
    'scikit-image==0.14.0',
    'pygame==1.9.2',
    'tflearn==0.3.2',
    'tqdm==4.28.1',
    'tensorflow==1.1.0',
    'h5py==2.8.0',
    'psutil==5.4.7',
    'Sphinx==1.7.9',
    'sphinx_rtd_theme==0.4.2',
    'sphinxcontrib-napoleon==0.7',
    'pyserial==3.4',
    'seaborn==0.9.0',
    'setuptools==40.2.0',
    'PyQt5==5.10',
    'cmocean==1.2'
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

class Documentation(distutils.cmd.Command):
    description = '''Build the documentation with Sphinx.
                   sphinx-apidoc is run for automatic generation of the sources.
                   sphinx-build then creates the html from these sources.'''
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        command = 'sphinx-apidoc -f -o docs/source pysilcam/ --separate'
        os.system(command)
        command = 'sphinx-build -b html ./docs/source ./docs/build'
        os.system(command)
        sys.exit()


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


setup(
    name='PySilCam',
    version='0.1.0',
    description='A Python interface to the SilCam.',
    long_description=read('README.md'),
    author='Emlyn Davies',
    author_email='emlyn.davies@sintef.no',
    install_requires=REQUIRES,
    # Use Python 3 branch on alternate repo for Pymba
    #dependency_links=['git+https://github.com/mabl/pymba@python3'],
    zip_safe=False,
    keywords='silcam',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['pysilcam'],
    entry_points={
        'console_scripts': [
            'silcam = pysilcam.__main__:silcam',
            'silcam-report = pysilcam.silcreport:silcreport',
        ],
        'gui_scripts': [
            'silcam-gui = pysilcam.silcamgui.silcamgui:main',
        ]
    },
    tests_require=['pytest'],
    cmdclass={'test': PyTest, 'build_sphinx': Documentation}
)
