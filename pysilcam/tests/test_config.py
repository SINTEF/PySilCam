# -*- coding: utf-8 -*-
import os
from pysilcam.config import load_config


def test_config_parser():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'config_example.ini')
    conf = load_config(filename)

    assert 'General' in conf
    assert 'Process' in conf
    assert conf['General']['version'] == '1'
