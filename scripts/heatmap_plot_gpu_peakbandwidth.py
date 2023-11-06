#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: wes
Created: Thu Sep 30 05:51:28 PDT 2021

Description: this code generates a 2D "heatmap" style plot using sample data that
is hard-coded into the code.

Inputs: none, all problem parameters are hard-coded.

Outputs: a plot showing the heatmap, displayed to the screen

Dependencies: matplotlib, numpy

Assumptions: Developed and Tested with Python 3.8.8 on MacOS 11.6
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

threads_per_block = ['32', '64', '128', '256', '512', '1024'] # y axis, 6 of them
thread_blocks = ["1", "4", "16", "64", "256", "1024", "4096"] # x axis, 7 of them

runtime = np.array([[0.01, 0.05, 0.17, 0.69, 2.55, 8.29, 16.34],
                    [0.03, 0.09, 0.33, 1.35, 4.49, 14.29, 25.60],
                    [0.05, 0.15, 0.65, 2.48, 8.29, 22.72, 28.17],
                    [0.08, 0.30, 1.17, 4.34, 14.18, 26.92, 28.64],
                    [0.13, 0.52, 1.96, 7.57, 21.44, 28.02, 28.26],
                    [0.23, 0.85, 3.31, 12.28, 23.71, 28.04, 26.70]])


fig, ax = plt.subplots(figsize=(9,6))
im = ax.imshow(runtime, cmap="coolwarm_r")

# We want to show all ticks...
ax.set_xticks(np.arange(len(thread_blocks)))
ax.set_yticks(np.arange(len(threads_per_block)))
# ... and label them with the respective list entries
ax.set_xticklabels(thread_blocks)
ax.set_yticklabels(threads_per_block)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(threads_per_block)): # y axis
    for j in range(len(thread_blocks)): # x axis
        text = ax.text(j, i, runtime[i, j],
                       ha="center", va="center", color="k")

ax.set_title("Peak Sustained Memory Bandwidth (%) on GPU-CUDA at Varying Block Size and Number of Blocks")
ax.set_ylabel('Threads per block')
ax.set_xlabel('Block Sizes')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("heatmap_gpu_PeakBandwidth.png")
plt.show()


# EOF
