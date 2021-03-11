# -*- coding: utf-8 -*-
'''
PySilCam configuration handling.
'''
import os
import configparser
import ast
from collections import namedtuple
import logging
import h5py
from pysilcam.pathadjust import PathLength

logger = logging.getLogger(__name__)

def load_config(filename):
    '''Load config file and validate content
    
    Args:
      filename (str) : filename including path

    Raises:
      RuntimeError : when file could not be read

    Returns:
      ConfigParser  : with the file parsed

    '''
    #Check that the file exists
    if not os.path.exists(filename):
        raise RuntimeError('Config file not found: {0}'.format(filename))
    
    ##Create ConfigParser and populate from file
    conf = configparser.ConfigParser()
    files_parsed = conf.read(filename)
    if filename not in files_parsed:
        raise RuntimeError('Could not parse config file {0}'.format(filename))

    return conf


def default_config_path():
    '''return the path to the default config file

    Returns:
        path_to_config (str)  : path to the default config file
    '''
    path = os.path.dirname(__file__)
    path_to_config = os.path.join(path, 'config_example.ini')
    return path_to_config

class PySilcamSettings:
    '''
    Class for SilCam settings
    '''
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
      filename (str)      : filename including path to camera config file
      config=None  (dict) : a dictionnary to store key-values. If config does not exist, an empty dict is created

    Returns:
      dict()              : with key value pairs of camera settings  
    
    '''

    if (config == None):
       config = dict()

    #Check that the file exists
    if (filename == None):
       return config

    filename = os.path.normpath(filename)

    if not os.path.exists(filename):
       logger.info('Camera config file not found: {0}'.format(filename))
       logger.debug('Camera config file not found: {0}'.format(filename))
       return config

    #Create SafeConfigParser and make it case sensitive
    config_parser = configparser.SafeConfigParser()
    config_parser.optionxform = str

    #Populate the parser from the file
    files_parsed = config_parser.read(filename)
    if filename not in files_parsed:
       logger.debug('Could not parse camera config file {0}'.format(filename))
       return config

    try:
       if not config_parser.has_section('Camera'):
          logger.debug('No Camera section in ini file: {0}'.format(filename))
          return config

       # File is read, find the camera section
       for k, v in config_parser.items('Camera'):
          try:
             parsed_val = ast.literal_eval(v)
          except:
             parsed_val = v
          config[k] = parsed_val

    except:
       logger.debug('Could not read camera config file: {0}'.format(filename))

    # return the configuration as a dict
    return config


def settings_from_h5(h5file):
    '''
    extracts PySilCamSettings from an exported hdf5 file created from silcam process

    :param h5file: created by pysilcam export functionality
    :return: Settings
    '''
    with h5py.File(h5file, 'r') as f:
        settings_dict = f['Meta'].attrs['Settings']
    cf = configparser.ConfigParser()
    cf.read_dict(ast.literal_eval(str(settings_dict)))
    Settings = PySilcamSettings(cf)

    return Settings


def updatePathLength(settings, logger):
    '''Adjusts the path length of systems with the actuator installed and RS232
    connected.

    Args:
        settings (PySilcamSettings): Settings read from a .ini file
                                     settings.logfile is optional
                                     settings.loglevel mest exist
        logger (logger object)     : logger object created using
                                     configure_logger()
    '''
    try:
        logger.info('Updating path length')
        pl = PathLength(settings.PostProcess.com_port)
        pl.gap_to_mm(settings.PostProcess.path_length)
        pl.finish()
    except:
        logger.warning('Could not open port. Path length will not be adjusted.')
