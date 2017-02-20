# -*- coding: utf-8 -*-
'''
PySilCam configuration handling.
'''
import configparser

# This is the required version of the configuration file
__configversion__ = 1


def load_config(filename):
    '''Load config file and validate content'''
    
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
