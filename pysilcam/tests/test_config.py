# -*- coding: utf-8 -*-
import os
from pysilcam.config import load_config, PySilcamSettings


def test_config_parser():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'config_example.ini')
    conf = load_config(filename)

    assert 'General' in conf
    assert conf['General']['version'] == '2'
    assert 'Background' in conf
    assert 'Process' in conf
    assert 'PostProcess' in conf
    assert 'ExportParticles' in conf
    assert 'NNClassify' in conf


def test_settings():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'config_example.ini')
    settings = PySilcamSettings(filename)

    assert hasattr(settings, 'General')
    assert hasattr(settings.General, 'version')
    assert hasattr(settings, 'Background')
    assert hasattr(settings, 'Process')
    assert hasattr(settings, 'PostProcess')
    assert hasattr(settings, 'ExportParticles')
    assert hasattr(settings, 'NNClassify')
