start C:\ProgramData\Anaconda3\envs\silcam\python.exe C:\Users\Eier\pysilcam\shell_tools\silc_monitor.py Z:\OHMSETT\data &

call C:\ProgramData\Anaconda3\Scripts\activate silcam

Z:
cd OHMSETT
silcam realtime config.ini data --discread --nomultiproc --appendstats
pause