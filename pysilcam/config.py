# -*- coding: utf-8 -*-
'''
PySilCam configuration handling.
'''
import os
import configparser
import ast
from collections import namedtuple
import logging

# This is the required version of the configuration file
__configversion__ = 1

def load_config(filename):
    '''Load config file and validate content'''

    #Check that the file exists
    if not os.path.exists(filename):
        raise RuntimeError('Config file not found: {0}'.format(filename))
    
    #Create ConfigParser and populate from file
    conf = configparser.ConfigParser()
    files_parsed = conf.read(filename)
    if filename not in files_parsed:
        raise RuntimeError('Could not parse config file {0}'.format(filename))

    #Check that we got the correction version
    fileversion = int(conf['General']['version'])
    expectversion = __configversion__
    if fileversion != expectversion:
        errmsg = 'Wrong configuration file version, expected {0}, got {1}.'
        raise RuntimeError(errmsg.format(fileversion, expectversion))

    return conf


class PySilcamSettings:
    def __init__(self, config):
        if isinstance(config, configparser.ConfigParser):
            self.config = config
        else:
            self.config = load_config(config)
        for sec in self.config.sections():
            cursec = dict()
            for k, v in self.config.items(sec):
                try:
                    parsed_val = ast.literal_eval(v)
                except:
                    parsed_val = v
                cursec[k] = parsed_val
            C = namedtuple(sec, cursec.keys())
            self.__dict__[sec] = C(**cursec)
