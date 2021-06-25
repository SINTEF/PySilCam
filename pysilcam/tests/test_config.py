# -*- coding: utf-8 -*-
import os
import re
import xml.etree.ElementTree as ET
from pysilcam.config import load_config, PySilcamSettings


def test_config_parser():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'config_example.ini')
    conf = load_config(filename)

    assert 'General' in conf
    assert 'Background' in conf
    assert 'Process' in conf
    assert 'PostProcess' in conf
    assert 'ExportParticles' in conf
    assert 'Camera' in conf
    assert 'NNClassify' in conf


def test_settings():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'config_example.ini')
    settings = PySilcamSettings(filename)

    assert hasattr(settings, 'General')
    assert hasattr(settings, 'Background')
    assert hasattr(settings, 'Process')
    assert hasattr(settings, 'PostProcess')
    assert hasattr(settings, 'ExportParticles')
    assert hasattr(settings, 'Camera')
    assert 'PixelFormat' not in settings.Camera  # Gives error due to Vimba
    assert hasattr(settings, 'NNClassify')


def test_camera_config_defaults():
    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'camera_config_defaults.xml')

    assert os.path.isfile(filename)

    # ET expects a single root, which our file doesn't have. So append:
    with open(filename) as f:
        xml = f.read()
    tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")

    CameraSettings = dict()
    # root/CameraSettings/FeatureGroup contains most settings:
    for child in tree[1][0]:
        CameraSettings[child.attrib['Name']] = child.text
    # Manually add 'TriggerSource': Value:
    CameraSettings[tree[1][1][4][1].attrib['Feature']] = tree[1][1][4][1].text
    # Manually add 'SyncOut1.SyncOutSource': Value:
    CameraSettings[tree[1][1][3][0].attrib['Feature']] = tree[1][1][3][0].text
    # Manually add 'SyncOut1.SyncOutPolarity': Value:
    CameraSettings[tree[1][1][3][1].attrib['Feature']] = tree[1][1][3][1].text

    assert CameraSettings['TriggerSource'] == 'FixedRate'
    assert CameraSettings['AcquisitionMode'] == 'Continuous'
    assert CameraSettings['PixelFormat'] == 'RGB8Packed'
    assert CameraSettings['StrobeDurationMode'] == 'Controlled'
    assert CameraSettings['StrobeSource'] == 'FrameTriggerReady'
    assert CameraSettings['SyncOutPolarity'] == 'Normal'
    assert CameraSettings['SyncOutSource'] == 'Strobe1'
