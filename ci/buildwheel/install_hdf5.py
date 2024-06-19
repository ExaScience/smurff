#!/usr/bin/env python3

import logging
from subprocess import check_call
import os
import shutil
from pathlib import Path
import h5py

logging.basicConfig(level=logging.INFO)

hdf5_basedir = Path(__file__).parent.absolute()
# hdf5_install_dir = hdf5_basedir / "hdf5"
hdf5_install_dir = Path("/usr/local/hdf5")

if hdf5_install_dir.exists():
    logging.warn(f"install dir already exists: {hdf5_install_dir} -- skipping")
    exit(0)

major, minor, release = h5py.h5.get_libversion()
full = f"{major}.{minor}.{release}"
logging.info(f"found h5py version: {full}")

h5py_package_dir = Path(h5py.__file__).parent
h5py_lib_dir = h5py_package_dir / '.dylibs'
logging.info(f"found h5py lib_dir: {h5py_lib_dir}")

hdf5_src_dir = hdf5_basedir / f"hdf5-{full}"

logging.info(f"building in : {hdf5_src_dir}")
logging.info(f"installing in : {hdf5_install_dir}")

url = f"https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-{major}.{minor}/hdf5-{full}/src/hdf5-{full}.tar.gz"
logging.info(f"download: {url}")
check_call(f"wget -q {url} -O - | tar -xzf -", shell=True, cwd=hdf5_basedir)

logging.info(f"build and install")
check_call(f"""
                ./configure --prefix={hdf5_install_dir} &&
                make -j &&
                sudo mkdir -m 777 {hdf5_install_dir} &&
                make install""",
            shell=True,
            cwd=hdf5_src_dir)

hdf5_lib_dir = hdf5_install_dir / "lib"
hdf5_lib_dir_backup = hdf5_install_dir / "lib.orig"

logging.info(f"replace dylibs in {hdf5_lib_dir} from {h5py_lib_dir}")
for f in h5py_lib_dir.glob('*.dylib'):
    logging.info(f"  {f} -> {h5py_lib_dir}/")
    shutil.copy(f, hdf5_lib_dir)





