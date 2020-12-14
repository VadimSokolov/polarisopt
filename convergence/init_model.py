#!/usr/bin/python
# Filename: run_convergence.py

import sys
import os
import shutil
from shutil import copyfile
from pathlib import Path

def init_model(data_directory):
    data_dir = Path(data_directory)
    if not data_dir.exists():
        print(f'ERROR: \'{data_directory}\' does not exist!')
        sys.exit(-1)

    backup_dir = data_dir / 'backup'
    if backup_dir.exists():
        print(f'WARNING: \'{backup_dir}\' already exists!')
        print(f'WARNING: Replacing backup directory!')
        shutil.rmtree(backup_dir)

    backup_dir.mkdir()
    for file in os.listdir(data_dir):
        if file.endswith(".sqlite"):
            shutil.copyfile(str(data_dir / file), str(backup_dir / file))

        if file.endswith("_skim_file.bin"):
            shutil.copyfile(str(data_dir / file), str(backup_dir / file))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage {sys.argv[0]} <data_directory>')
        sys.exit(-1)

    init_model(sys.argv[1])