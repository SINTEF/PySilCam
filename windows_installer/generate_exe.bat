echo Generating executable for PySilcam GUI
call pip install pyinstaller
call pip install pywin32
call conda install -y pathlib
call pip install distutils
call python ../setup.py develop

echo Run pyinstaller. The exe file generated is located in corecalculator/dist
call pyinstaller -F silcamgui.spec --log-level=DEBUG
