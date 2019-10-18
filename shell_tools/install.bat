cd ..\
conda env create --force environment.yml & ^
activate silcam & ^
python setup.py develop & ^
python setup.py test


