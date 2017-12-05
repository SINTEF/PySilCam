# -*- coding: utf-8 -*-
'''
PySilCam configuration handling.
'''
import os
import configparser
import ast
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

# This is the required version of the configuration file
__configversion__ = 2

def load_config(filename):
    '''Load config file and validate content'''
    #Check that the file exists
    if not os.path.exists(filename):
        raise RuntimeError('Config file not found: {0}'.format(filename))
    
    ##Create ConfigParser and populate from file
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
                    # Try to fix problem with parsing \
                    v = v.replace('\\','\\\\')
                    try: #again
                       parsed_val = ast.literal_eval(v)
                    except:
                       parsed_val = v

                cursec[k] = parsed_val
            C = namedtuple(sec, cursec.keys())
            self.__dict__[sec] = C(**cursec)

def load_camera_config(filename, config=None):
    '''Load camera config file and validate content

    Args:
      filename (str)  : file name of the config file
                        do nothing if the file could not be read
      config   (dict) : a dictionary to add key-values to
                        creates an ampty dict if its None

    Returns:
      config  (dict) : 
      
    '''

    if (config == None):
      config = dict()

    #Check that the file exists
    if (filename == None):
        return config

    filename = os.path.normpath(filename)

    if not os.path.exists(filename):
        print('Camera config file not found: {0}'.format(filename))
        logger.debug('Camera config file not found: {0}'.format(filename))
        return config

    #Create SafeConfigParser and make it case sensitive
    config_parser = configparser.SafeConfigParser()
    config_parser.optionxform = str;

    #Populate the parser from the file
    files_parsed = config_parser.read(filename)
    if filename not in files_parsed:
        logger.debug('Could not parse camera config file {0}'.format(filename))
        return config

    try:
       if not config_parser.has_section('Camera'):
          logger.debug('No Camera section in ini file:', filename)
          return config

       # File is read, find the camera section
       for k, v in config_parser.items('Camera'):
         try:
            parsed_val = ast.literal_eval(v)
         except:
            parsed_val = v
         config[k] = parsed_val

    except:
      logger.debug('Could not read camera config file:', filename)

    # return the configuration as a dict
    return config
