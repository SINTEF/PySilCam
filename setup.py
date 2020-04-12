# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
import distutils.cmd


class PyTestNoSkip(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        # use pytest plugin to force error if a test is skipped
        self.test_args = ['--error-for-skips']
        self.test_suite = True

    def run_tests(self):
        import pytest
        params = {"args":self.test_args}
        params["args"] +=  ["--junitxml", "test-report/output.xml"]
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


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
    cmdclass={'test': PyTest,
              'test_noskip': PyTestNoSkip,
              'build_sphinx': Documentation}
)
