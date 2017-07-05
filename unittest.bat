@echo off
REM Default installs
activate sctest && conda install tensorflow -c conda-forge && pip install tflearn && python setup.py test
