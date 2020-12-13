#!/usr/bin/python
# Filename: CSV_Utilities.py

#import shutil
import sys
#import os
#import subprocess
#from shutil import copyfile
from pathlib import Path
#import json
#import sqlite3
import csv
#import traceback


def append_column(src, tgt, column, header_text):
    with src.open('r') as csvinput:
        with tgt.open('w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append(header_text)
            all.append(row)

            for row in reader:
                row.append(row[column])
                all.append(row)

            writer.writerows(all)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage %s <source_file> <target_file>' % (sys.argv[0]))
        sys.exit(-1)

    src_file = Path(sys.argv[1])
    if not src_file.exists():
        print('ERROR: Source File %s does not exist!' % src_file)
        sys.exit(-1)

    tgt_file = Path(sys.argv[2])
    if not tgt_file.exists():
        print('ERROR: Target File %s does not exist!' % tgt_file)
        sys.exit(-1)

    append_column(src_file, tgt_file, 1, "time")
    append_column(src_file, tgt_file, 4, str(src_file.name))
