# -*- mode: python -*-

from pathlib import Path
from distutils.sysconfig import get_python_lib

block_cipher = None

# Some files are required by some python packages but are not included in the package.
# They need to be added manually.
# If this is not included in the build, a
# No such file or directory: 'C:\\Users\\jorgenk\\AppData\\Local\\Temp\\_MEI183602\\distributed\\config.yaml'
# will be raised
site_packages_path = Path(get_python_lib())

cmocean = site_packages_path / "cmocean" / "rgb" / "thermal-rgb.txt"

def get_pandas_path():
    """Helper function to get the path the pandas lib."""
    import pandas

    return pandas.__path__[0]

pandas_tree = Tree(get_pandas_path(), prefix="pandas", excludes=["*.pyc"])

a = Analysis(
    ["../pysilcam/silcamgui/silcamgui.py"],
    pathex=[],
    binaries=[],
    # path to file, where to put it in final app
    datas=[(str(cmocean), "thermal-rgb.txt")],
    hiddenimports=["scipy._lib.messagestream"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

a.datas += pandas_tree
a.binaries = filter(lambda x: "pandas" not in x[0], a.binaries)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='windows_silcamgui',
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,
)