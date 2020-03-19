#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : getting_started.py
# description     : Demonstration of PyPyP basics.
# author          : Guillaume Dumas
# date            : 2020-03-18
# version         : 1
# python_version  : 3.7
# ==============================================================================

import os
import mne
from hypyp.viz import transform
from hypyp.viz import plot_sensors_2d, plot_links_2d
from hypyp.viz import plot_sensors_3d, plot_links_3d
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading data files & extracting sensor infos
epo1 = mne.read_epochs(os.path.join("data", "subject1-epo.fif"), preload=True)
loc1 = copy(np.array([ch['loc'][:3] for ch in epo1.info['chs']]))
lab1 = [ch + "_1" for ch in epo1.ch_names]

epo2 = mne.read_epochs(os.path.join("data", "subject2-epo.fif"), preload=True)
loc2 = copy(np.array([ch['loc'][:3] for ch in epo2.info['chs']]))
lab2 = [ch + "_2" for ch in epo2.ch_names]
loc2 = transform(loc2)

# Visualization of inter-brain connectivity in 2D
plt.figure(figsize=(10, 20))
plt.gca().set_aspect('equal', 'box')
plt.axis('off')
plot_sensors_2d(loc1, loc2, lab1, lab2)
plot_links_2d(loc1, loc2, C=np.random.rand(len(loc1), len(loc2)), threshold=0.95, steps=10)
plt.tight_layout()
plt.show()

# Visualization of inter-brain connectivity in 3D
loc2 = transform(loc2, traY=0.15, rotZ=0)

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.axis('off')
plot_sensors_3d(ax, loc1, loc2, lab1, lab2)
plot_links_3d(ax, loc1, loc2, C=np.random.rand(len(loc1), len(loc2)), threshold=0.95, steps=10)
plt.tight_layout()
plt.show()