# -*- coding: utf-8 -*-
import unittest
import unittest.mock as mock
from subprocess import check_output


def test_echo():
    '''An example test.'''
    result = run_cmd("echo hello world")
    assert result == "hello world\n"


def run_cmd(cmd):
    '''Run a shell command `cmd` and return its output.'''
    return check_output(cmd, shell=True).decode('utf-8')


# Mock up the whole pymba module for testing offline
@mock.patch.dict('sys.modules', pymba=mock.Mock())
def test_aquire():
    import pysilcam.aquire
