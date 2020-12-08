To test the GUI:

1) Download and unzip the test data from:
https://pysilcam.blob.core.windows.net/test-data/pysilcam-testdata.zip
   
2) For a basic launch of the GUI: 
```bash
python setup.py develop
silcam-gui
```

3) Use the 'Browse' button to navigate to pysilcam-testdata/unittest-data/STN04 and 'select folder'

4) Press 'Load config file'. When the file query window arrives, navigate up a directory level to the 'unittest-data' folder, select 'config.ini'

5) Press 'START'

6) If you want to, press 'Live raw' during analysis to see some images (STN04 is not oil & gas data, so the data itself might look strange)