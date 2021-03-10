# -*- mode: python -*-

from pathlib import Path
from distutils.sysconfig import get_python_lib
import os
import cmocean
import matplotlib as mpl

block_cipher = None

# Manual specification of cmocean and matplotlib data dirs, add these to PyInstaller 'datas'
cmocean_data_dir = os.path.join(cmocean.__path__[0], 'rgb')
mpl_data_dir = os.path.join(mpl.__path__[0], 'mpl-data')
datas = [(cmocean_data_dir, 'cmocean/rgb'), (mpl_data_dir, 'matplotlib/mpl-data')]


def get_pandas_path():
    """Helper function to get the path the pandas lib."""
    import pandas

    return pandas.__path__[0]

pandas_tree = Tree(get_pandas_path(), prefix="pandas", excludes=["*.pyc"])

a = Analysis(
    ["../pysilcam/silcamgui/silcamgui.py"],
    pathex=[],
    binaries=[],
    datas=datas,
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
    bootloader_ignore_signals=False,
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True
)
