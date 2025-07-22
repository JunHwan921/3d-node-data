# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# 필요한 모든 패키지 수집
packages = ['numpy', 'sklearn', 'scipy', 'PyQt5', 'pyqtgraph', 'OpenGL']
datas = []
binaries = []
hiddenimports = []

for package in packages:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# src 폴더 추가
datas += [('src', 'src')]

# 추가 hidden imports
hiddenimports += [
    'numpy.core._multiarray_umath',
    'numpy.core._multiarray_tests',
    'numpy.random._common',
    'numpy.random._bounded_integers',
    'numpy.random._mt19937',
    'numpy.random.bit_generator',
    'numpy.random._pickle',
    'numpy.random._pcg64',
    'numpy.random._philox',
    'numpy.random._sfc64',
    'numpy.random._generator',
    'numpy.random.mtrand',
    'scipy._lib.messagestream',
    'scipy.spatial.transform._rotation_groups',
    'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
    'sklearn.utils._weight_vector',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NodeEditor3D',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 일단 콘솔 표시로 오류 확인
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NodeEditor3D',
)
