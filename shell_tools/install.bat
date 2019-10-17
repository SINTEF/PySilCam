cd ..\
conda env create --force --yes environment.yml & ^
activate silcam & ^
python setup.py develop & ^
python setup.py test


