# -*- coding: utf-8 -*-
import os
from pysilcam.config import load_config, PySilcamSettings, load_camera_config

   
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

def test_camera_config_parser():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'camera_config_defaults.ini')
    filename = os.path.normpath(filename)
    conf = load_camera_config(filename)

    assert conf.get('configversion') != None
    assert 'configversion' in conf
    assert conf['configversion'] == 1

    assert 'TriggerSource' in conf
    assert 'AcquisitionMode' in conf
    assert 'ExposureTimeAbs' in conf
    assert 'PixelFormat' in conf
    assert 'StrobeDuration' in conf
    assert 'StrobeDelay' in conf
    assert 'StrobeDurationMode' in conf
    assert 'StrobeSource' in conf
    assert 'SyncOutPolarity' in conf
    assert 'SyncOutSelector' in conf
    assert 'SyncOutSource' in conf

