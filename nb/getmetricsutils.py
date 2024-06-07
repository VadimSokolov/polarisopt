import pandas as pd
import numpy as np
import h5py
import json
import glob
import os
import sqlite3
import h5py
import matplotlib.pyplot as plt
from concurrent import futures
from itertools import repeat

# EDA
def plotsum(f):
    sum = pd.read_csv(f'{f}/summary.csv', index_col=False)['in_network'].values
    with h5py.File(f'{f}/Austin-Result.h5', 'r') as f:
        # ref_output = f['link_moe']['link_travel_time'][:]*f['link_moe']['link_in_volume'][:]
        ref_output = f['link_moe']['num_vehicles_in_link'][:]
        ref_output =  np.sum(ref_output, axis=1)
    plt.close()
    fig, axs = plt.subplots(1,2,figsize=(6,2))
    axs[0].plot(sum)
    axs[1].plot(ref_output)

def getnet(fl):
    with h5py.File(f'{fl}/Austin-Result.h5', 'r') as f:
        ref_output = f['link_moe']['link_travel_time'][:]*f['link_moe']['link_in_volume'][:]
        # moe = f['link_moe']['num_vehicles_in_link'][:]
    moe =  np.sum(ref_output, axis=1)
    return(moe)

def pltnet(f):
    moe = getnet(f)
    plt.plot(moe)