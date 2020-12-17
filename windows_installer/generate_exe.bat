rem run this script from the top pysilcam directory .\windows_installer\generate_exe.bet

echo Generating executable for corecalculator

rem C:\Users\emlynd\Anaconda3\Scripts\activate.bat

rem C:\Users\emlynd\PySilCam

rem delete existing exe file (if it already exists)
rem if exist "%CD%"\dist\ccrunner.exe del "%CD%"\dist\ccrunner.exe

rem create conda environment and activate it.
rem call conda env create --force -f environment.yml
rem C:\Users\emlynd\Anaconda3\Scripts\activate.bat  C:\Users\emlynd\Anaconda3\envs\silcam
call activate silcam
rem If the conda environment cannot be set, there is no need to continue
rem if %errorlevel% neq 0 exit /b %errorlevel%

rem install pyinstaller
call pip install pyinstaller
rem pypiwin32
call pip install pywin32
call conda install -y pathlib
call pip install distutils

rem install corecalculator

rem C:\Users\emlynd\PySilCam

call python setup.py develop

cd windows_installer
rem run pyinstaller. The exe file generated is located in corecalculator/dist
call pyinstaller -F silcamgui.spec --log-level=DEBUG

pause